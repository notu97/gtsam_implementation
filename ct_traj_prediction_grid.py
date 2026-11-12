import gtsam
import numpy as np
from typing import List

from gtsam import Pose3, StateData
from gtsam.symbol_shorthand import X, V, L, Y, W
import time
import os
import argparse

from collections import defaultdict

# ---------- dataset selection (defaults reproduce the original MH01 run in cwd) ----------
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", default=".", help="folder with the ORB-SLAM3 dump files")
_ap.add_argument("--seq", default="MH01", help="sequence tag used in file names, e.g. MH01")
_ap.add_argument("--out-dir", default=".", help="where to write the gp_*.txt results")
_ap.add_argument("--state-dt", type=float, default=0.5, help="spacing of GP states [s]")
ARGS = _ap.parse_args()
os.makedirs(ARGS.out_dir, exist_ok=True)

def data_path(prefix):   # e.g. data_path("kf_dataset") -> <data-dir>/kf_dataset-MH01_mono.txt
    return os.path.join(ARGS.data_dir, f"{prefix}-{ARGS.seq}_mono.txt" if prefix != "Frame_obs"
                        else f"Frame_obs_{ARGS.seq}_mono.txt")

def out_path(prefix):    # e.g. out_path("gp_grid") -> <out-dir>/gp_grid_MH01_mono.txt
    return os.path.join(ARGS.out_dir, f"{prefix}_{ARGS.seq}_mono.txt")


TOL_NS = 1_000_000        # 1 ms timestamp matching tolerance
FRAME_STRIDE = 4          # use every 4th non-KF frame to start
MIN_OBS = 3               # min observations per landmark
MAX_REPROJ_PX = 5.0       # init outlier threshold
STATE_DT = ARGS.state_dt  # GP state spacing [s]
FRAME_DT = 0.05           # EuRoC camera period (20 Hz)

# EuRoC cam0 intrinsics. ORB-SLAM3's EuRoC yaml resizes images 752x480 -> 600x350
# (Camera.newWidth/newHeight), so the pixels in the dumps live in the resized image.
# Pixels are already undistorted (mvKeysUn), so no distortion model here.
SX, SY = 600 / 752, 350 / 480
K = gtsam.Cal3_S2(458.654 * SX, 457.296 * SY, 0.0, 367.215 * SX, 248.375 * SY)

def s_to_ns(s):            # "1403636579.963555574"
    a, b = s.split('.')
    return int(a) * 10**9 + int(b.ljust(9, '0')[:9])

def ns_to_s(ns):           # 1403636579963555574 -> "1403636579.963555574"
    ns = int(ns)
    sign = '-' if ns < 0 else ''
    a, b = divmod(abs(ns), 10**9)
    return f"{sign}{a}.{b:09d}"


def load_kf_dataset(path):
    """Parse a TUM file with lines: t_ns tx ty tz qx qy qz qw -> (int64 ns array, [Pose3])."""
    times_ns, poses = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith('#'):
                continue
            times_ns.append(int(tok[0].split('.')[0]))
            tx, ty, tz, qx, qy, qz, qw = map(float, tok[1:8])
            poses.append(Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(tx, ty, tz)))
    return np.array(times_ns, dtype=np.int64), poses


def nearest(ts_sorted, q):
    i = np.searchsorted(ts_sorted, q)
    cand = [j for j in (i - 1, i) if 0 <= j < len(ts_sorted)]
    j = min(cand, key=lambda j: abs(int(ts_sorted[j]) - q))
    return j, abs(int(ts_sorted[j]) - q)


# ---------- load landmarks + keyframe observations ----------
def load_mapPoint_and_kfObs(path, kf_t):
    P = {}
    obs_kf = []                                 # (lid, kf_index, u, v)
    for line in open(path):
        s = line.split()
        if s[0] == "P":
            P[int(s[1])] = np.array(list(map(float, s[2:5])))
        else:
            k, dt = nearest(kf_t, s_to_ns(s[2]))
            if dt <= TOL_NS:
                obs_kf.append((int(s[1]), k, float(s[3]), float(s[4])))
    return P, obs_kf


# ---------- load non-keyframe observations ----------
def load_non_keyframe_obs(path, kf_t, fr_t, landmarks_id):
    kf_set_frames = set()
    for t in kf_t:
        j, dt = nearest(fr_t, int(t))
        if dt <= TOL_NS:
            kf_set_frames.add(j)

    obs_fr = []                                 # (lid, frame_index, u, v)
    for line in open(path):
        s = line.split()
        lid = int(s[0])
        if lid not in landmarks_id:
            continue                            # culled / merged later
        j, dt = nearest(fr_t, s_to_ns(s[1]))
        if dt > TOL_NS or j in kf_set_frames or j % FRAME_STRIDE:
            continue
        obs_fr.append((lid, j, float(s[2]), float(s[3])))
    return obs_fr


# ---------- reprojection check at initialization ----------
def reproj_err(T_WC, p, u, v):
    cam = gtsam.PinholeCameraCal3_S2(T_WC, K)
    try:
        uv = cam.project(gtsam.Point3(*p))  # raises on point behind camera
    except RuntimeError:
        return np.inf
    return float(np.hypot(uv[0] - u, uv[1] - v))


def save_tum(path, times_ns, poses):
    with open(path, "w") as f:
        for t, T in zip(times_ns, poses):
            p = T.translation()
            q = T.rotation().toQuaternion()
            f.write(f"{int(t)} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{q.x():.9f} {q.y():.9f} {q.z():.9f} {q.w():.9f}\n")


# ======================= 1. Load and filter (unchanged) =======================
kf_times_ns, kf_poses = load_kf_dataset(data_path("kf_dataset"))
f_times_ns, f_poses = load_kf_dataset(data_path("f_dataset"))

P, obs_kf = load_mapPoint_and_kfObs(data_path("MapPoints_KF_UVobs_dataset"), kf_times_ns)
obs_fr = load_non_keyframe_obs(data_path("Frame_obs"), kf_times_ns, f_times_ns, P)

good_kf = [o for o in obs_kf if reproj_err(kf_poses[o[1]], P[o[0]], o[2], o[3]) < MAX_REPROJ_PX]
good_fr = [o for o in obs_fr if reproj_err(f_poses[o[1]], P[o[0]], o[2], o[3]) < MAX_REPROJ_PX]


# ======================= 2. Put every observation on a frame index =======================
# Keyframe observations become ordinary frame observations (keyframes are frames).
obs_all = list(good_fr)                         # (lid, frame_index, u, v)
for lid, k, u, v in good_kf:
    j, dt = nearest(f_times_ns, int(kf_times_ns[k]))
    if dt <= TOL_NS:                            # the first KF predates the frame file: dropped
        obs_all.append((lid, j, u, v))

count = defaultdict(int)
for o in obs_all:
    count[o[0]] += 1
keep = {lid for lid, c in count.items() if c >= MIN_OBS}
obs_all = [o for o in obs_all if o[0] in keep]


# ======================= 3. Uniform grid of GP states =======================
# Time relative to the first frame (seconds) keeps full double precision.
t0_ns = f_times_ns[0]
f_times = (f_times_ns - t0_ns) * 1e-9

step = max(1, int(round(STATE_DT / FRAME_DT)))  # 0.5 s -> every 10th frame
state_js = list(range(0, len(f_times), step))   # frame index of each state
if state_js[-1] != len(f_times) - 1:
    state_js.append(len(f_times) - 1)           # last frame is a state, so all frames are bracketed
state_t = f_times[state_js]
states: List[StateData] = [StateData(X(i), V(i), state_t[i]) for i in range(len(state_js))]
state_of_frame = {j: i for i, j in enumerate(state_js)}

print(f"{len(states)} GP states every {STATE_DT} s ({step} frames), "
      f"{len(keep)} landmarks, {len(obs_all)} observations")


def interp_state(j):
    """Virtual GP state at frame j. Y(j), W(j) are NOT variables: they never go into `initial`."""
    return StateData(Y(j), W(j), f_times[j])


def bracket(tau):
    """Grid states (i, i+1) with t_i < tau < t_{i+1}."""
    i = int(np.searchsorted(state_t, tau)) - 1
    return states[i], states[i + 1]


# ---------- initial values: ORB-SLAM3 frame poses at the state frames ----------
initial = gtsam.Values()
init_T = [f_poses[j] for j in state_js]
for i, T in enumerate(init_T):
    initial.insert(X(i), T)
for i, j in enumerate(state_js):                # velocity from neighbouring FRAMES (50 ms apart)
    a, b = max(j - 1, 0), min(j + 1, len(f_poses) - 1)
    w = gtsam.Pose3.Logmap(f_poses[a].between(f_poses[b])) / (f_times[b] - f_times[a])
    initial.insert(V(i), w)                     # 6-vector [rot; trans], body frame

for lid in keep:
    initial.insert(L(lid), gtsam.Point3(*P[lid]))


# ======================= 4. Factors =======================
Qc = 0.1 * np.ones(6)
graph = gtsam.NonlinearFactorGraph()

# WNOA motion priors between consecutive grid states
for i in range(1, len(states)):
    graph.add(gtsam.WnoaMotionFactorPose3(states[i - 1], states[i], Qc))

# Gauge (X0) and scale (translation of X1)
graph.add(gtsam.PriorFactorPose3(X(0), init_T[0], gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)))
graph.add(gtsam.PriorFactorPose3(X(1), init_T[1],
                                 gtsam.noiseModel.Diagonal.Sigmas([1e3] * 3 + [1e-4] * 3)))

noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, 1.0))
# Huber inside WnoaInterpFactorPose3 segfaults in GTSAM 4.3.0 -> plain Gaussian there
noise_interp = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

n_plain = n_interp = 0
for lid, j, u, v in obs_all:
    if j in state_of_frame:                     # observation at a state's frame: plain factor
        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(u, v), noise, X(state_of_frame[j]), L(lid), K))
        n_plain += 1
    else:                                       # between states: interpolated factor
        s0, s1 = bracket(f_times[j])
        proj = gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(u, v), noise_interp, Y(j), L(lid), K)
        graph.add(gtsam.WnoaInterpFactorPose3(proj, {s0, s1}, {interp_state(j)}, Qc))
        n_interp += 1
print(f"Added {len(states) - 1} WNOA factors, {n_plain} plain and {n_interp} "
      f"interpolated reprojection factors")


# ======================= 5. Optimize =======================
params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(100)
params.setRelativeErrorTol(1e-6)
params.setAbsoluteErrorTol(1e-6)
params.setVerbosityLM("SUMMARY")

print(f"Factors: {graph.size()}, variables: {initial.size()}")
print(f"Initial error: {graph.error(initial):.1f}")
t_start = time.time()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()
print(f"Final error:   {graph.error(result):.1f}  "
      f"({optimizer.iterations()} iters, {time.time() - t_start:.1f} s)")


# ======================= 6. Query T(t) at every frame and save =======================
query_js = [j for j in range(len(f_times)) if j not in state_of_frame]
interp_vals = gtsam.updateInterpValuesPose3(
    graph, result, set(states), {interp_state(j) for j in query_js}, Qc)

def pose_at(j):
    return result.atPose3(X(state_of_frame[j])) if j in state_of_frame else interp_vals.atPose3(Y(j))

save_tum(out_path("gp_grid_states"), [f_times_ns[j] for j in state_js],
         [result.atPose3(X(i)) for i in range(len(states))])
save_tum(out_path("gp_grid_full"), f_times_ns, [pose_at(j) for j in range(len(f_times))])
print(f"Saved {out_path('gp_grid_states')} and {out_path('gp_grid_full')}")
