import matplotlib.pyplot as plt
from tracker.multi_sim import run_multi_demo

def main():
    # Set clutter_per_step=0 for a clean README plot.
    X1, X2, measurements, hist = run_multi_demo(seed=0, n=250, dt=0.1, clutter_per_step=0)

    plt.figure(figsize=(9, 7))
    plt.plot(X1[:, 0], X1[:, 1], label="GT target 1", linewidth=2)
    plt.plot(X2[:, 0], X2[:, 1], label="GT target 2", linewidth=2)

    # If clutter is enabled, you can show a subset of measurements
    for k in range(0, len(measurements), 10):
        xs = [p[0] for p in measurements[k]]
        ys = [p[1] for p in measurements[k]]
        plt.scatter(xs, ys, s=10, alpha=0.15, color="gray")

    # Only plot tracks that survive long enough (reduces clutter in legend/plot)
    min_len = 20
    for tid, pts in hist.items():
        if len(pts) < min_len:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, linewidth=2, label=f"track {tid}")

    plt.title("Multi-target tracking (Mahalanobis gating + greedy NN association)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)
    plt.show()

if __name__ == "__main__":
    main()