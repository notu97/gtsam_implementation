"""
List every frame used by ct_traj_prediction_grid.py, per sequence.

Runs sections 1-3 of ct_traj_prediction_grid.py verbatim (load, 5 px filter, MIN_OBS,
state grid) and stops before the factors/optimization, so the list is exactly what the
optimizer saw. Each frame is mapped to its cam0 image in the EuRoC mav0 folder.

Usage:
  ~/.venv-gtsam/bin/python list_grid_frames.py --seqs MH01 MH03 MH04

Output: frame_lists/frames_used_<SEQ>.csv with one row per used frame:
  frame_idx      index in f_dataset-<SEQ>_mono.txt (the script's j)
  cam0_t_ns      exact cam0 timestamp (mav0/cam0/data.csv)
  cam0_file      image file name in mav0/cam0/data/
  role           state       : GP state X(i) (every 10th frame + last frame)
                 interp_meas : measurement frame between states (WnoaInterpFactorPose3)
  is_state, is_keyframe, n_obs (reprojection observations kept on that frame)
plus frame_lists/summary.txt.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_DIRS = {"MH01": "MH_01_easy", "MH03": "MH_03_medium", "MH04": "MH_04_difficult"}


def grid_sections_1_to_3(seq, state_dt):
    """Execute the grid script up to its factor section; return its namespace."""
    src = open(os.path.join(HERE, "ct_traj_prediction_grid.py")).read()
    src = src.split("# ======================= 4. Factors")[0]
    argv = sys.argv
    sys.argv = ["ct_traj_prediction_grid.py", "--data-dir", os.path.join(HERE, f"MH_{seq[2:]}_data"),
                "--seq", seq, "--out-dir", os.path.join(HERE, "frame_lists", ".tmp"),
                "--state-dt", str(state_dt)]
    ns = {"__name__": "grid_frames"}
    try:
        exec(compile(src, "ct_traj_prediction_grid.py", "exec"), ns)
    finally:
        sys.argv = argv
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["MH01", "MH03", "MH04"])
    ap.add_argument("--euroc-dir", default=os.path.expanduser("~/Dataset/euroc"))
    ap.add_argument("--state-dt", type=float, default=0.5)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "frame_lists"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    summary = []
    for seq in a.seqs:
        g = grid_sections_1_to_3(seq, a.state_dt)
        f_t = g["f_times_ns"]
        n_frames = len(f_t)
        states = set(g["state_js"])

        n_obs = np.zeros(n_frames, dtype=int)
        for _, j, _, _ in g["obs_all"]:
            n_obs[j] += 1

        kf_frames = set()
        for t in g["kf_times_ns"]:
            j, dt = g["nearest"](f_t, int(t))
            if dt <= g["TOL_NS"]:
                kf_frames.add(j)

        used = sorted(states | set(np.nonzero(n_obs)[0].tolist()))

        # map to the exact cam0 image (ORB-SLAM3 stamps are float-rounded by a few hundred ns)
        d = np.genfromtxt(os.path.join(a.euroc_dir, SEQ_DIRS[seq], "mav0", "cam0", "data.csv"),
                          delimiter=",", comments="#", dtype=str)
        cam_t = np.array([int(x) for x in d[:, 0]], dtype=np.int64)
        cam_f = [x.strip() for x in d[:, 1]]

        out = os.path.join(a.out_dir, f"frames_used_{seq}.csv")
        n_state = n_interp = 0
        max_dt = 0
        with open(out, "w") as fh:
            fh.write("frame_idx,cam0_t_ns,cam0_file,role,is_state,is_keyframe,n_obs\n")
            for j in used:
                c, dt = g["nearest"](cam_t, int(f_t[j]))
                max_dt = max(max_dt, dt)
                role = "state" if j in states else "interp_meas"
                n_state += role == "state"
                n_interp += role == "interp_meas"
                fh.write(f"{j},{cam_t[c]},{cam_f[c]},{role},{int(j in states)},"
                         f"{int(j in kf_frames)},{n_obs[j]}\n")
        line = (f"{seq}: {len(used)} of {n_frames} ORB-SLAM3 frames used ({100 * len(used) / n_frames:.1f}%) "
                f"| {n_state} state frames, {n_interp} interpolated measurement frames "
                f"| {len(kf_frames & set(used))} are keyframes "
                f"| {int((n_obs[list(states)] == 0).sum())} state frames have no observations "
                f"| {len(cam_t)} images in mav0/cam0 | max ORB->cam0 stamp offset {max_dt} ns")
        print(line, flush=True)
        summary.append(line)

    with open(os.path.join(a.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
