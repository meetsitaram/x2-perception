"""Desktop: render perception_view.py's npz: the chest depth the stack got, the student's fine height map and the planner scan
with ACP's contact targets, for the last frame (or --frame). Only the stack's own inputs and outputs, nothing else.

    python3 render_perception.py x.npz out.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES, X0, NX, NY = 0.025, -0.40, 52, 44          # models/acp_depth/fine_student.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--frame", type=int, default=-1)
    a = ap.parse_args()
    d = np.load(a.npz)
    k = a.frame
    fine, plan, tg, occ, z = d["fine"][k], d["plan"][k], d["targets"][k], d["occ"][k], d["depth"][k]
    floor = float(np.median(fine))              # the map is height minus root height; shown relative to its median (the floor)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    i0 = ax[0].imshow(np.where(z > 0, z, np.nan), cmap="jet_r", vmin=0.3, vmax=3.5)
    ax[0].set_title("chest depth into the stack (m, white = no return)"); ax[0].axis("off"); plt.colorbar(i0, ax=ax[0], shrink=0.7)
    ext = [(NY / 2) * RES, -(NY / 2) * RES, X0, X0 + NX * RES]          # left at left, forward up
    i1 = ax[1].imshow(fine - floor, origin="lower", extent=ext, cmap="viridis", vmin=-0.05, vmax=0.2)
    ax[1].set_title("student fine map: height above floor (m)"); ax[1].set_xlabel("left (m)"); ax[1].set_ylabel("forward (m)")
    plt.colorbar(i1, ax=ax[1], shrink=0.7)
    gx = -0.8 + 0.1 * np.arange(17); gy = 0.1 * (np.arange(11) - 5)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    i2 = ax[2].scatter(Y.ravel(), X.ravel(), c=plan - np.median(plan), cmap="viridis", vmin=-0.05, vmax=0.2, s=90, marker="s")
    on = occ > 0.5
    ax[2].scatter(tg[on, 1], tg[on, 0], s=160, facecolors="none", edgecolors="r", lw=2, label="contact targets (occ > 0.5)")
    ax[2].set_xlim(0.6, -0.6); ax[2].set_ylim(-0.9, 0.9); ax[2].set_aspect("equal"); ax[2].legend(loc="lower right", fontsize=8)
    ax[2].set_title("planner scan 17 x 11 (height above floor) + targets"); ax[2].set_xlabel("left (m)"); ax[2].set_ylabel("forward (m)")
    plt.colorbar(i2, ax=ax[2], shrink=0.7)
    for x in ax[1:]:
        x.plot(0, 0, "w^", ms=10)
    fig.suptitle("deployed ACP stack on the live robot data (frame %d of %d, camera offset pitch %+.2f roll %+.2f deg)" % (
        k % len(d["fine"]), len(d["fine"]), *d["cam"]))
    plt.tight_layout()
    plt.savefig(a.out, dpi=90)


if __name__ == "__main__":
    main()
