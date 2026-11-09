import gtsam
import numpy as np
from typing import List

from gtsam import Pose3, StateData
from gtsam.symbol_shorthand import X, V, L

from collections import defaultdict

# Plotting tools: Needs pyvista
# from gtsam.examples.GaussianProcessWnoaInterpolationSE3Helpers import collect_estimated_plot_data, collect_interp_plot_data, plot_results



TOL_NS = 1_000_000        # 1 ms timestamp matching tolerance
FRAME_STRIDE = 4          # use every 4th non-KF frame to start
MIN_OBS = 3               # min observations per landmark
MAX_REPROJ_PX = 5.0       # init outlier threshold

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

def ns_str_to_ns(s):                       # "1403636579913555712.000000"
    return int(s.split(".")[0])

def load_kf_dataset(path):
    """Parse a keyframe file with lines: t_ns tx ty tz qx qy qz qw.

    Returns (kf_times_ns, kf_poses) where kf_times_ns is a list of int ns
    timestamps and kf_poses is a list of gtsam.Pose3.
    """
    kf_times_ns, kf_poses = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith('#'):
                continue
            t_ns = int(tok[0].split('.')[0])        # "1403636579763555840.000000"
            tx, ty, tz, qx, qy, qz, qw = map(float, tok[1:8])
            pose = Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz),
                         gtsam.Point3(tx, ty, tz))
            kf_times_ns.append(t_ns)
            kf_poses.append(pose)
    return kf_times_ns, kf_poses



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

    return P , obs_kf
 
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


kf_times_ns, kf_poses = load_kf_dataset("kf_dataset-MH01_mono.txt")
f_times_ns, f_poses = load_kf_dataset("f_dataset-MH01_mono.txt")

P, obs_kf = load_mapPoint_and_kfObs("MapPoints_KF_UVobs_dataset-MH01_mono.txt", kf_times_ns)
# print(" Some Landmark Id: ", P.keys())
obs_fr = load_non_keyframe_obs("Frame_obs_MH01_mono.txt", kf_times_ns, f_times_ns, P)

good_kf = [o for o in obs_kf if reproj_err(kf_poses[o[1]], P[o[0]], o[2], o[3]) < MAX_REPROJ_PX]
good_fr = [o for o in obs_fr if reproj_err(f_poses[o[1]], P[o[0]], o[2], o[3]) < MAX_REPROJ_PX]
 
 
# ---------- keep landmarks with enough observations ----------
count = defaultdict(int)
for o in good_kf + good_fr:
    count[o[0]] += 1
keep = {lid for lid, c in count.items() if c >= MIN_OBS}
good_kf = [o for o in good_kf if o[0] in keep]
good_fr = [o for o in good_fr if o[0] in keep]


# Use time relative to the first keyframe (in seconds) so the double keeps
# full precision; absolute epoch seconds would lose sub-microsecond digits.
t0_ns = kf_times_ns[0]
kf_times = [float(ns_to_s(t - t0_ns)) for t in kf_times_ns]

kf_states: List[StateData] = [
    StateData(X(k), V(k), kf_times[k]) for k in range(len(kf_times))
]

# Sanity check: first/last few keyframe states
print(f"Loaded {len(kf_states)} keyframe states, t0 = {ns_to_s(t0_ns)} s")
print(f"{'k':>4} {'pose key':>8} {'vel key':>8} {'t [s]':>14} {'dt [s]':>10} {'abs time [s]':>22}  position")
for k in list(range(5)) + list(range(len(kf_states) - 3, len(kf_states))):
    s = kf_states[k]
    dt = s.time - kf_states[k - 1].time if k > 0 else 0.0
    p = kf_poses[k].translation()
    print(f"{k:>4} {gtsam.Symbol(s.pose).string():>8} {gtsam.Symbol(s.velocity).string():>8} "
          f"{s.time:>14.9f} {dt:>10.6f} {ns_to_s(kf_times_ns[k]):>22}  "
          f"[{p[0]: .4f} {p[1]: .4f} {p[2]: .4f}]")
dts = np.diff(kf_times)
print(f"dt min/mean/max = {dts.min():.4f} / {dts.mean():.4f} / {dts.max():.4f} s, "
      f"monotonic: {bool(np.all(dts > 0))}")


# 

# --- WNOA motion priors between consecutive keyframe states ---
# Power Spectral Density for the GP prior (diagonal-only, one entry per twist dim)
Qc = 0.001 * np.ones(6)

graph = gtsam.NonlinearFactorGraph()
for k in range(1, len(kf_states)):
    graph.add(gtsam.WnoaMotionFactorPose3(kf_states[k - 1], kf_states[k], Qc))

print(f"Added {graph.size()} WNOA motion factors (Qc = {Qc})")


# ---------- insert landmark variables ----------
initial = gtsam.Values()
for lid in keep:
    initial.insert(L(lid), gtsam.Point3(*P[lid]))

print(" Inserted good Landmark variables")
e = np.array([reproj_err(kf_poses[o[1]], P[o[0]], o[2], o[3]) for o in good_kf])
print(f"landmarks in P        : {len(P)}")
print(f"landmarks kept        : {len(keep)}")
print(f"KF observations       : {len(obs_kf)} -> {len(good_kf)} after filtering")
print(f"non-KF observations   : {len(obs_fr)} -> {len(good_fr)} after filtering")
print(f"KF reproj RMSE (px)   : {np.sqrt(np.mean(e**2)):.3f}")
