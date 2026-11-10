import argparse

import matplotlib
import matplotlib.pyplot as plt
from tracker.multi_sim import run_multi_demo

def main():
    ap = argparse.ArgumentParser(description="Seeded multi-target tracking demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clutter", type=int, default=0, help="false alarms per step")
    ap.add_argument("--save", help="write the plot to this path instead of showing it")
    args = ap.parse_args()
    if args.save:
        matplotlib.use("Agg")
    # Set clutter_per_step=0 for a clean README plot.
    X1, X2, measurements, hist = run_multi_demo(seed=args.seed, n=250, dt=0.1, clutter_per_step=args.clutter)

    plt.figure(figsize=(9, 7))
    plt.plot(X1[:, 0], X1[:, 1], "k--", label="ground truth", linewidth=1.5, zorder=3)
    plt.plot(X2[:, 0], X2[:, 1], "k--", linewidth=1.5, zorder=3)

    # If clutter is enabled, you can show a subset of measurements
    for k in range(0, len(measurements), 10):
        xs = [p[0] for p in measurements[k]]
        ys = [p[1] for p in measurements[k]]
        plt.scatter(xs, ys, s=10, alpha=0.15, color="gray")

    # Only plot tracks that survive long enough (reduces clutter in legend/plot)
    min_len = 50  # short-lived tracks are mostly clutter-initiated
    for tid, pts in hist.items():
        if len(pts) < min_len:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, linewidth=2, label=f"track {tid}")

    plt.title(f"Multi-target tracking (seed {args.seed}, clutter {args.clutter}/step)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)
    if args.save:
        plt.savefig(args.save, dpi=110, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()