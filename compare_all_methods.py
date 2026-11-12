"""
Apples-to-apples comparison on EuRoC cam0:
    ORB-SLAM3  |  Keyframe GP  |  Grid GP  |  VGGT-Omega

Sub-commands
------------
  run-vggt   Run VGGT-Omega (../run_vggt_omega_euroc.py) on each sequence.
             Needs a Python with vggt_omega + CUDA (default: the MonoGS conda env).
  eval       Evaluate all four methods with one protocol (needs numpy/scipy/matplotlib).

Example
-------
  python compare_all_methods.py run-vggt --seqs MH01 MH03 MH04
  python compare_all_methods.py eval     --seqs MH01 MH03 MH04

Protocol (identical for every method)
-------------------------------------
* Ground truth: EuRoC state_groundtruth_estimate0 (IMU body) moved to cam0 with T_BC,
  p_WC = p_WB + R_WB t_BC  (same as plot_ate_cam0_frame.py). Nearest GT sample, <= 5 ms.
* Time association: every estimated pose is snapped to its cam0 image (nearest
  cam0/data.csv timestamp, <= 2 ms). This removes the float rounding in the ORB-SLAM3
  dumps (e.g. ...555840 vs the true ...555584) and VGGT's seconds-format stamps.
* Same frames: the main table uses ONLY the cam0 images for which ALL methods have a
  pose and GT exists (intersection). VGGT-Omega runs on every --stride'th image, so this
  is effectively the VGGT frame set, minus frames outside the keyframe GP's span.
* Same alignment: one Sim(3) Umeyama fit per method per sequence on those frames
  (positions only; monocular scale is free for every method).
* Metrics: ATE RMSE / median / max [m], translational RPE over --rpe-dt seconds [m],
  rotation error after alignment [deg], alignment scale.
* A second table scores each method on ALL of its own frames (not comparable across
  methods, for reference only).

Inputs (defaults match this repo's layout)
------------------------------------------
  ORB-SLAM3    MH_XX_data/f_dataset-MHXX_mono.txt                (every tracked frame)
  Keyframe GP  results_comparison/MHXX/kf/gp_full_MHXX_mono.txt
  Grid GP      results_comparison/MHXX/grid/gp_grid_full_MHXX_mono.txt
  VGGT-Omega   results_vggt/MHXX/<run folder>/traj_est_tum.txt
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

SEQ_DIRS = {"MH01": "MH_01_easy", "MH02": "MH_02_easy", "MH03": "MH_03_medium",
            "MH04": "MH_04_difficult", "MH05": "MH_05_difficult"}

# EuRoC cam0 extrinsics (mav0/cam0/sensor.yaml, T_BS): camera -> body
T_BC = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                 [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                 [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                 [0.0, 0.0, 0.0, 1.0]])

METHODS = ["ORB-SLAM3", "Keyframe GP", "Grid GP", "VGGT-Omega"]
COLORS = {"ORB-SLAM3": "tab:blue", "Keyframe GP": "tab:orange",
          "Grid GP": "tab:green", "VGGT-Omega": "tab:red"}


def mav0(euroc_dir, seq):
    return os.path.join(euroc_dir, SEQ_DIRS[seq], "mav0")


# ============================================================================ run-vggt
def cmd_run_vggt(a):
    for seq in a.seqs:
        out_base = os.path.join(a.vggt_dir, seq, "run")      # script appends a settings tag
        cmd = [a.vggt_python, os.path.join(PROJ, "run_vggt_omega_euroc.py"),
               "--dataset", mav0(a.euroc_dir, seq), "--checkpoint", a.checkpoint,
               "--stride", str(a.stride), "--chunk", str(a.chunk), "--overlap", str(a.overlap),
               "--resolution", str(a.resolution), "--out", out_base, "--undistort"]
        if a.max_size:
            cmd.append("--max-size")
        os.makedirs(os.path.dirname(out_base), exist_ok=True)
        print(f"[{seq}] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=PROJ)
        wall = time.time() - t0
        run_dir = newest_vggt_run(a.vggt_dir, seq)
        json.dump({"wall_time_s": wall}, open(os.path.join(run_dir, "wall_time.json"), "w"))
        print(f"[{seq}] done in {wall:.0f} s -> {run_dir}", flush=True)


def newest_vggt_run(vggt_dir, seq):
    # layouts: <vggt_dir>/<SEQ>/run_*/ (run-vggt) or <vggt_dir>/<SEQ>_*/ (run_vggt_omega_framelist.py)
    cands = glob.glob(os.path.join(vggt_dir, seq, "run_*")) + glob.glob(os.path.join(vggt_dir, f"{seq}_*"))
    runs = [d for d in cands if os.path.exists(os.path.join(d, "traj_est_tum.txt"))]
    if not runs:
        raise FileNotFoundError(f"No VGGT-Omega run with traj_est_tum.txt under {vggt_dir}/{seq}")
    return max(runs, key=os.path.getmtime)


# ============================================================================ loading
def stamp_to_ns(tok):
    """'1403636579913555712.000000' (ns) or '1403636579.763555765' (s) -> int ns, exactly."""
    a, _, b = tok.partition(".")
    if len(a) >= 16:                       # already nanoseconds
        return int(a)
    return int(a) * 10**9 + int(b.ljust(9, "0")[:9])


def load_tum(path):
    """TUM file -> (t_ns int64 [N], p [N,3], R_wc [N,3,3]), sorted by time."""
    from scipy.spatial.transform import Rotation as Rot
    t, p, q = [], [], []
    for line in open(path):
        s = line.split()
        if not s or s[0].startswith("#"):
            continue
        t.append(stamp_to_ns(s[0]))
        p.append([float(v) for v in s[1:4]])
        q.append([float(v) for v in s[4:8]])                 # qx qy qz qw
    t = np.array(t, dtype=np.int64)
    o = np.argsort(t)
    return t[o], np.array(p)[o], Rot.from_quat(np.array(q)[o]).as_matrix()


def load_cam0_stamps(euroc_dir, seq):
    d = np.genfromtxt(os.path.join(mav0(euroc_dir, seq), "cam0", "data.csv"),
                      delimiter=",", comments="#", dtype=str)
    return np.array([int(x) for x in d[:, 0]], dtype=np.int64)


def load_gt_cam0(euroc_dir, seq):
    """EuRoC GT (IMU body) -> cam0 pose: p_WC = p_WB + R_WB t_BC, R_WC = R_WB R_BC."""
    from scipy.spatial.transform import Rotation as Rot
    d = np.genfromtxt(os.path.join(mav0(euroc_dir, seq), "state_groundtruth_estimate0", "data.csv"),
                      delimiter=",", comments="#")
    R_WB = Rot.from_quat(d[:, [5, 6, 7, 4]]).as_matrix()     # file: qw qx qy qz
    p = d[:, 1:4] + np.einsum("nij,j->ni", R_WB, T_BC[:3, 3])
    return d[:, 0].astype(np.int64), p, R_WB @ T_BC[:3, :3]


def nearest(ts_sorted, q, tol_ns):
    i = np.clip(np.searchsorted(ts_sorted, q), 1, len(ts_sorted) - 1)
    j = np.where(np.abs(q - ts_sorted[i - 1]) <= np.abs(q - ts_sorted[i]), i - 1, i)
    return j, np.abs(ts_sorted[j] - q) <= tol_ns


def to_frame_index(traj, cam_t, tol_ns=2_000_000):
    """Snap each pose to its cam0 image index -> {frame_idx: (p, R)}."""
    t, p, R = traj
    j, ok = nearest(cam_t, t, tol_ns)
    return {int(k): (p[n], R[n]) for n, k in enumerate(j) if ok[n]}


# ============================================================================ metrics
def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def score(frames, est, gt_p, gt_R, t_sec, rpe_dt):
    """frames: sorted frame indices; est: {idx: (p, R)}; gt_*: aligned to `frames`."""
    P = np.array([est[f][0] for f in frames])
    Rw = np.array([est[f][1] for f in frames])
    s, Ra, ta = umeyama_sim3(P, gt_p)
    Pa = (s * (Ra @ P.T)).T + ta
    e = np.linalg.norm(Pa - gt_p, axis=1)

    k = np.searchsorted(t_sec, t_sec + rpe_dt)               # pair each frame with one rpe_dt later
    i = np.nonzero(k < len(t_sec))[0]
    rpe = np.linalg.norm((Pa[k[i]] - Pa[i]) - (gt_p[k[i]] - gt_p[i]), axis=1)

    dR = np.einsum("nji,njk->nik", gt_R, np.einsum("ij,njk->nik", Ra, Rw))   # R_gt^T (Ra R_est)
    rot = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))

    return dict(n=len(frames), ate_rmse=float(np.sqrt(np.mean(e ** 2))), ate_median=float(np.median(e)),
                ate_max=float(e.max()), rpe_rmse=float(np.sqrt(np.mean(rpe ** 2))) if len(i) else float("nan"),
                rot_rmse_deg=float(np.sqrt(np.mean(rot ** 2))), scale=float(s),
                _t=t_sec, _p=Pa, _e=e)


def evaluate_on(frames, trajs, cam_t, gt, rpe_dt):
    gt_t, gt_p, gt_R = gt
    j, ok = nearest(gt_t, cam_t[frames], 5_000_000)
    frames, j = np.asarray(frames)[ok], j[ok]
    t_sec = (cam_t[frames] - cam_t[frames[0]]) * 1e-9
    return {m: score(frames, trajs[m], gt_p[j], gt_R[j], t_sec, rpe_dt) for m in trajs}, gt_p[j]


# ============================================================================ eval
def method_paths(a, seq):
    s = seq[2:]
    return {"ORB-SLAM3": os.path.join(HERE, f"MH_{s}_data", f"f_dataset-{seq}_mono.txt"),
            "Keyframe GP": os.path.join(a.gp_dir, seq, "kf", f"gp_full_{seq}_mono.txt"),
            "Grid GP": os.path.join(a.gp_dir, seq, "grid", f"gp_grid_full_{seq}_mono.txt"),
            "VGGT-Omega": os.path.join(newest_vggt_run(a.vggt_dir, seq), "traj_est_tum.txt")}


def cmd_eval(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out_dir, exist_ok=True)
    common_tab, own_tab, sources = {}, {}, {}
    for seq in a.seqs:
        cam_t = load_cam0_stamps(a.euroc_dir, seq)
        gt = load_gt_cam0(a.euroc_dir, seq)
        paths = method_paths(a, seq)
        sources[seq] = {m: os.path.relpath(p, HERE) for m, p in paths.items()}
        trajs = {m: to_frame_index(load_tum(paths[m]), cam_t) for m in METHODS}

        common = sorted(set.intersection(*(set(v) for v in trajs.values())))
        res, P_gt = evaluate_on(common, trajs, cam_t, gt, a.rpe_dt)
        common_tab[seq] = {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in res.items()}
        own_tab[seq] = {}
        for m in METHODS:                                    # reference: each on its own frames
            r, _ = evaluate_on(sorted(trajs[m]), {m: trajs[m]}, cam_t, gt, a.rpe_dt)
            own_tab[seq][m] = {k: v for k, v in r[m].items() if not k.startswith("_")}

        n_eval = common_tab[seq]["ORB-SLAM3"]["n"]           # common frames that also have GT
        print(f"\n=== {seq}: {n_eval} common frames with GT "
              f"(ORB {len(trajs['ORB-SLAM3'])}, KF-GP {len(trajs['Keyframe GP'])}, "
              f"Grid-GP {len(trajs['Grid GP'])}, VGGT {len(trajs['VGGT-Omega'])})")
        for m in METHODS:
            r = common_tab[seq][m]
            print(f"  {m:12s} ATE {r['ate_rmse']:.4f} m (median {r['ate_median']:.4f}, max {r['ate_max']:.4f})  "
                  f"RPE {r['rpe_rmse']:.4f} m  rot {r['rot_rmse_deg']:.2f} deg  scale {r['scale']:.3f}")

        # ---- per-sequence figures: top-down + XZ + error over time.
        # VGGT-Omega is an order of magnitude worse, so it gets its own figure; the other
        # three share one so their differences stay visible. A combined one is kept too.
        groups = {f"ate_{seq}_orb_gp.png": ["ORB-SLAM3", "Keyframe GP", "Grid GP"],
                  f"ate_{seq}_vggt.png": ["VGGT-Omega"],
                  f"ate_{seq}.png": METHODS}
        for fname, ms in groups.items():
            fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
            for c, (i0, i1, name) in enumerate([(0, 1, "top-down (x-y)"), (0, 2, "side (x-z)")]):
                ax[c].plot(P_gt[:, i0], P_gt[:, i1], "k", lw=1.6, label="Ground truth")
                for m in ms:
                    ax[c].plot(res[m]["_p"][:, i0], res[m]["_p"][:, i1], lw=1, color=COLORS[m], label=m)
                ax[c].set(xlabel=f"{'xyz'[i0]} [m]", ylabel=f"{'xyz'[i1]} [m]", title=f"{seq}: {name}")
                ax[c].set_aspect("equal", adjustable="datalim"); ax[c].grid(alpha=.3)
            ax[0].legend(fontsize=8)
            for m in ms:
                ax[2].plot(res[m]["_t"], res[m]["_e"], lw=1, color=COLORS[m],
                           label=f"{m}: RMSE {res[m]['ate_rmse']:.3f} m")
            ax[2].set(xlabel="time [s]", ylabel="translation error [m]",
                      title=f"{seq}: ATE on {n_eval} common frames"); ax[2].grid(alpha=.3)
            ax[2].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, fname), dpi=150); plt.close(fig)

    # ---- summary bar chart (log scale: VGGT is an order of magnitude apart)
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(a.seqs)), 0.8 / len(METHODS)
    for k, m in enumerate(METHODS):
        v = [common_tab[s][m]["ate_rmse"] for s in a.seqs]
        b = ax.bar(x + (k - (len(METHODS) - 1) / 2) * w, v, w, color=COLORS[m], label=m)
        ax.bar_label(b, [f"{u:.3f}" for u in v], fontsize=7, padding=2)
    ax.set_xticks(x, a.seqs); ax.set_ylabel("ATE RMSE [m] (common frames)"); ax.set_yscale("log")
    ax.grid(axis="y", alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "ate_summary.png"), dpi=150); plt.close(fig)

    # ---- tables
    json.dump({"common_frames": common_tab, "own_frames": own_tab, "sources": sources,
               "rpe_dt_s": a.rpe_dt}, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write("# ORB-SLAM3 vs Keyframe GP vs Grid GP vs VGGT-Omega (EuRoC cam0)\n\n")
        f.write("Main table: all methods on the SAME frames (intersection), GT moved to cam0, "
                "one Sim(3) per method.\n\n")
        for key, title in [("ate_rmse", "ATE RMSE [m]"), ("ate_max", "ATE max [m]"),
                           ("rpe_rmse", f"RPE over {a.rpe_dt:g} s [m]"),
                           ("rot_rmse_deg", "Rotation error RMSE [deg]")]:
            f.write(f"### {title}\n\n| Seq | N | " + " | ".join(METHODS) + " |\n|---|---|" + "---|" * len(METHODS) + "\n")
            for s in a.seqs:
                vals = [common_tab[s][m][key] for m in METHODS]
                best = int(np.nanargmin(vals))
                cells = [f"**{v:.4f}**" if i == best else f"{v:.4f}" for i, v in enumerate(vals)]
                f.write(f"| {s} | {common_tab[s]['ORB-SLAM3']['n']} | " + " | ".join(cells) + " |\n")
            f.write("\n")
        f.write("### Reference: each method on ALL of its own frames (not comparable across methods)\n\n"
                "| Seq | " + " | ".join(f"{m} ATE (N)" for m in METHODS) + " |\n|---|" + "---|" * len(METHODS) + "\n")
        for s in a.seqs:
            f.write(f"| {s} | " + " | ".join(f"{own_tab[s][m]['ate_rmse']:.4f} ({own_tab[s][m]['n']})"
                                            for m in METHODS) + " |\n")
        f.write("\n### Sources\n\n")
        for s in a.seqs:
            for m in METHODS:
                f.write(f"- {s} {m}: `{sources[s][m]}`\n")
    print(f"\nSaved results.md, results.json, ate_<seq>.png, ate_summary.png to {a.out_dir}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seqs", nargs="+", default=["MH01", "MH03", "MH04"])
        p.add_argument("--euroc-dir", default=os.path.expanduser("~/Dataset/euroc"))
        p.add_argument("--vggt-dir", default=os.path.join(HERE, "results_vggt"))

    r = sub.add_parser("run-vggt", help="run VGGT-Omega via ../run_vggt_omega_euroc.py")
    common(r)
    r.add_argument("--vggt-python", default=os.path.expanduser("~/anaconda3/envs/MonoGS/bin/python"))
    r.add_argument("--checkpoint", default=os.path.join(PROJ, "checkpoints", "vggt_omega_1b_512.pt"))
    r.add_argument("--stride", type=int, default=5, help="every Nth cam0 image (20 Hz source)")
    r.add_argument("--chunk", type=int, default=60)
    r.add_argument("--overlap", type=int, default=20)
    r.add_argument("--resolution", type=int, default=512)
    r.add_argument("--no-max-size", dest="max_size", action="store_false",
                   help="default uses --max-size (the best MH01 setting so far)")
    r.set_defaults(func=cmd_run_vggt)

    e = sub.add_parser("eval", help="evaluate all four methods with one protocol")
    common(e)
    e.add_argument("--gp-dir", default=os.path.join(HERE, "results_comparison"),
                   help="folder with <SEQ>/kf/gp_full_*.txt and <SEQ>/grid/gp_grid_full_*.txt")
    e.add_argument("--rpe-dt", type=float, default=1.0)
    e.add_argument("--out-dir", default=os.path.join(HERE, "results_all_methods"))
    e.set_defaults(func=cmd_eval)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
