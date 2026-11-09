import gtsam
import numpy as np
from typing import List

from gtsam import Pose3, StateData
from gtsam.symbol_shorthand import X, V, L
import time

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
    return np.array(kf_times_ns, dtype=np.int64), kf_poses



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
# kf_times = [float(ns_to_s(t - t0_ns)) for t in kf_times_ns]

kf_times = (kf_times_ns - t0_ns) * 1e-9  

kf_states: List[StateData] = [
    StateData(X(k), V(k), kf_times[k]) for k in range(len(kf_times))
]

# Set priors for Pose and Velocity
initial = gtsam.Values()

for k, T in enumerate(kf_poses):
    initial.insert(X(k), T)
for k in range(len(kf_poses)):
    a, b = max(k - 1, 0), min(k + 1, len(kf_poses) - 1)
    dt = kf_times[b] - kf_times[a]
    w = gtsam.Pose3.Logmap(kf_poses[a].between(kf_poses[b])) / dt
    initial.insert(V(k), w)          # 6-vector [rot; trans], body frame

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
Qc = 0.1 * np.ones(6)

graph = gtsam.NonlinearFactorGraph()
for k in range(1, len(kf_states)):
    graph.add(gtsam.WnoaMotionFactorPose3(kf_states[k - 1], kf_states[k], Qc))

print(f"Added {graph.size()} WNOA motion factors (Qc = {Qc})")

graph.add(gtsam.PriorFactorPose3(X(0), kf_poses[0],
             gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)))
graph.add(gtsam.PriorFactorPose3(X(1), kf_poses[1],
            gtsam.noiseModel.Diagonal.Sigmas([1e3]*3 + [1e-4]*3)))  # fixes translation → scale

# # ---------- insert landmark variables ----------

# for lid in keep:
#     initial.insert(L(lid), gtsam.Point3(*P[lid]))

# ---------- [Without Interp factors] insert landmark variables ----------
# KF-only run: keep landmarks with >= 2 keyframe observations
kf_count = defaultdict(int)
for o in good_kf:
    kf_count[o[0]] += 1

for lid in keep:
    if kf_count[lid] >= 2:
        initial.insert(L(lid), gtsam.Point3(*P[lid]))

good_kf = [o for o in good_kf if kf_count[o[0]] >= 2]
# --------------------------------------------------


print(" Inserted good Landmark variables")
e = np.array([reproj_err(kf_poses[o[1]], P[o[0]], o[2], o[3]) for o in good_kf])
print(f"landmarks in P        : {len(P)}")
print(f"landmarks kept        : {len(keep)}")
print(f"KF observations       : {len(obs_kf)} -> {len(good_kf)} after filtering")
print(f"non-KF observations   : {len(obs_fr)} -> {len(good_fr)} after filtering")
print(f"KF reproj RMSE (px)   : {np.sqrt(np.mean(e**2)):.3f}")


# Add Plain reprojection factor on the Keyframe observations
noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, 1.0))

r'''

GenericProjectionFactorCal3_S2 Explanation:

For each entry (lid, k, u, v) in good_kf (landmark lid was seen in keyframe k at pixel $(u, v)$), 
it creates a factor that:

- Takes the current estimates of the camera pose X(k) ($T_{WC}$) and the landmark L(lid) ($p_\ell$).
- Projects the landmark into the image with intrinsics K:
      $$\hat u = \pi\!\left(K,\ T_{WC}^{-1}\, p_\ell\right)$$

- Compares the projection with the measured pixel:
      $$e = (u, v) - \hat u$$

- Weights the error with noise ($\sigma = 1$ px, Huber kernel).

During optimization, GTSAM adjusts X(k) and L(lid) to make these pixel errors small, 
balanced against the WNOA priors. Over all 139k factors, this is bundle adjustment on the keyframes. 
In short, these factors tie the camera poses to the 3D points through what the camera actually saw.'''

for lid, k, u, v in good_kf:
    graph.add(gtsam.GenericProjectionFactorCal3_S2(
        gtsam.Point2(u, v), noise, X(k), L(lid), K))


params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(100)
params.setRelativeErrorTol(1e-6)
params.setAbsoluteErrorTol(1e-6)
params.setVerbosityLM("SUMMARY")          # prints cost per iteration
 
print(f"Factors: {graph.size()}, variables: {initial.size()}")
print(f"Initial error: {graph.error(initial):.1f}")
 
t_start = time.time()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()
print(f"Final error:   {graph.error(result):.1f}  "
      f"({optimizer.iterations()} iters, {time.time() - t_start:.1f} s)")
 
 
# ---------- final reprojection RMSE on keyframe observations ----------
def reproj_err_result(lid, k, u, v):
    cam = gtsam.PinholeCameraCal3_S2(result.atPose3(X(k)), K)
    try:
        uv = cam.project(result.atPoint3(L(lid)))
    except RuntimeError:                  # point ended up behind the camera
        return np.inf
    return np.hypot(uv[0] - u, uv[1] - v)
 
e_opt = np.array([reproj_err_result(*o) for o in good_kf])
n_bad = int(np.sum(~np.isfinite(e_opt)))
e_fin = e_opt[np.isfinite(e_opt)]
print(f"KF reproj RMSE after opt (px): {np.sqrt(np.mean(e_fin**2)):.3f}  "
      f"(median {np.median(e_fin):.3f}, {n_bad} obs behind camera)")
 
 
# ---------- save optimized keyframe trajectory (TUM format, T_WC, t in ns) ----------
def save_tum(path, times_ns, poses):
    with open(path, "w") as f:
        for t, T in zip(times_ns, poses):
            p = T.translation()
            q = T.rotation().toQuaternion()       # gtsam Quaternion: w(), x(), y(), z()
            f.write(f"{int(t)} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{q.x():.9f} {q.y():.9f} {q.z():.9f} {q.w():.9f}\n")
 
opt_poses = [result.atPose3(X(k)) for k in range(len(kf_poses))]
save_tum("gp_kf_MH01_mono.txt", kf_times_ns, opt_poses)
print("Saved gp_kf_MH01_mono.txt")