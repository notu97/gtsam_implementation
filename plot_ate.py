"""
Sim(3)-aligned ATE: GP result vs ORB-SLAM3 vs EuRoC ground truth.

Usage:
  python plot_ate.py <GT data.csv> <traj1.txt> [<traj2.txt> ...]
  e.g. python plot_ate.py mav0/state_groundtruth_estimate0/data.csv \
           kf_dataset-MH01_mono.txt gp_kf_MH01_mono.txt

Trajectory files: TUM format, first column = timestamp in ns (int or "....000000").
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_gt(path):
    d = np.genfromtxt(path, delimiter=",", comments="#")
    return d[:, 0].astype(np.int64), d[:, 1:4]          # t_ns, p_RS_R (x,y,z)


def load_tum(path):
    t, p = [], []
    for line in open(path):
        s = line.split()
        if not s or s[0].startswith("#"):
            continue
        t.append(int(s[0].split(".")[0]))
        p.append([float(v) for v in s[1:4]])
    return np.array(t, dtype=np.int64), np.array(p)


def associate(t_est, t_gt, tol_ns=5_000_000):           # 5 ms
    idx = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    left, right = t_gt[idx - 1], t_gt[idx]
    j = np.where(np.abs(t_est - left) <= np.abs(t_est - right), idx - 1, idx)
    ok = np.abs(t_gt[j] - t_est) <= tol_ns
    return np.nonzero(ok)[0], j[ok]


def umeyama_sim3(src, dst):
    """Find s, R, t minimizing ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(path, t_gt, p_gt):
    t, p = load_tum(path)
    i, j = associate(t, t_gt)
    s, R, tr = umeyama_sim3(p[i], p_gt[j])
    p_al = (s * (R @ p[i].T)).T + tr
    err = np.linalg.norm(p_al - p_gt[j], axis=1)
    return dict(name=path, t=(t[i] - t_gt[0]) * 1e-9, p=p_al, err=err,
                rmse=np.sqrt(np.mean(err**2)), scale=s)


def main():
    args = sys.argv[1:]
    out_png = "ate_comparison.png"
    if "-o" in args:                                    # optional: -o <output.png>
        i = args.index("-o")
        out_png = args[i + 1]
        del args[i:i + 2]
    t_gt, p_gt = load_gt(args[0])
    runs = [ate(f, t_gt, p_gt) for f in args[1:]]

    for r in runs:
        print(f"{r['name']:40s} ATE RMSE = {r['rmse']:.4f} m   scale = {r['scale']:.3f}   "
              f"N = {len(r['err'])}")

    fig = plt.figure(figsize=(16, 5))
    ax3 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132)
    axe = fig.add_subplot(133)

    ax3.plot(*p_gt.T, "k", lw=1, label="Ground truth")
    ax2.plot(p_gt[:, 0], p_gt[:, 1], "k", lw=1, label="Ground truth")
    for r in runs:
        lbl = f"{r['name']} ({r['rmse']:.4f} m)"
        ax3.plot(*r["p"].T, lw=1, label=lbl)
        ax2.plot(r["p"][:, 0], r["p"][:, 1], lw=1, label=lbl)
        axe.plot(r["t"], r["err"], lw=1, label=lbl)

    ax3.set(title="3D trajectory", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax2.set(title="Top-down (XY)", xlabel="x [m]", ylabel="y [m]")
    ax2.axis("equal"); ax2.grid(alpha=0.3)
    axe.set(title="ATE over time", xlabel="time [s]", ylabel="translation error [m]")
    axe.grid(alpha=0.3)
    for a in (ax3, ax2, axe):
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
