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




kf_times_ns, kf_poses = load_kf_dataset("kf_dataset-MH01_mono.txt")

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