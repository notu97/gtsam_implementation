import gtsam
import numpy as np
from typing import List

from gtsam import Pose3, StateData
from gtsam.symbol_shorthand import X, V

# Plotting tools: Needs pyvista
# from gtsam.examples.GaussianProcessWnoaInterpolationSE3Helpers import collect_estimated_plot_data, collect_interp_plot_data, plot_results


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


TOL_NS = 1_000_000        # 1 ms timestamp matching tolerance

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
 


kf_times_ns, kf_poses = load_kf_dataset("kf_dataset-MH01_mono.txt")
f_times_ns, f_poses = load_kf_dataset("f_dataset-MH01_mono.txt")

P, obs_kf = load_mapPoint_and_kfObs("MapPoints_KF_UVobs_dataset-MH01_mono.txt", kf_times_ns)


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
