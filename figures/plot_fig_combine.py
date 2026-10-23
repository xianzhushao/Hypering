#!/usr/bin/env python3
"""single-column: (a) test embeddings on the Poincare disk (PCA angle,
exact geodesic radius), (b) normalised ring gaps over 10 seeds.
Usage: python plot_fig_combine.py [disk_data_dir] [ring_radii.csv] [out_stem]
       (defaults: data/disk_embedding, data/ring_radii_10seeds.csv, fig_combine_pca)"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _redistribute import redistribute_angles_per_class   # copied verbatim from plot_results.py

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "disk_embedding")
F3 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "ring_radii_10seeds.csv")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "fig_combine_pca")
NAMES = ["CN", "SMC", "EMCI", "LMCI", "AD"]
COLS = ["#E0B416", "#6CC24A", "#21908D", "#3B528B", "#440154"]
ACC, INK, MUTED, GUIDE = "#D9642A", "#262626", "#7F7F7F", "#D4D4D4"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "axes.edgecolor": "#404040",
                     "xtick.color": "#404040", "ytick.color": "#404040",
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})

# ---------- data (a)
X = pd.read_csv(os.path.join(D2, "test_location.csv"), header=None).values
y = pd.read_csv(os.path.join(D2, "test_true_pred.csv"), header=None).values[:, 0].astype(int)
radii = pd.read_csv(os.path.join(D2, "radii.csv"), header=None).values.ravel()[:5]
EPS = 1e-5
xn = np.clip(np.linalg.norm(X, axis=1, keepdims=True), EPS, None)
tan = np.arctanh(np.clip(xn, 0, 1 - EPS)) * X / xn                      # log_0 (c = 1)
d = 2 * np.arctanh(np.clip(xn.ravel(), 0, 1 - EPS))                     # geodesic distance to origin
xy = PCA(n_components=2).fit_transform(tan)
theta = redistribute_angles_per_class(np.arctan2(xy[:, 1], xy[:, 0]), d, y, radii, 5,
                                      mode="stretch", min_frac=0.35)
# ---------- data (b)
df = pd.read_csv(F3)
R = df[[f"r_{s}" for s in NAMES]].to_numpy(float)
G = np.diff(R, axis=1); S = G / G.sum(1, keepdims=True)

fig = plt.figure(figsize=(3.45, 2.0))
axa = fig.add_axes([0.0, 0.13, 0.5, 0.80])
axb = fig.add_axes([0.62, 0.25, 0.36, 0.65])

# ---------- (a) disk
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import LinearSegmentedColormap
t = np.linspace(0, 2 * np.pi, 400)
lim = max(d.max(), radii.max()) * 1.06
# hyperbolic backdrop: shading deepens toward the boundary
gx, gy = np.meshgrid(np.linspace(-lim, lim, 500), np.linspace(-lim, lim, 500))
rr = np.hypot(gx, gy) / lim
cm = LinearSegmentedColormap.from_list("bg", ["#FFFFFF", "#F3F4F8", "#E3E6EF"])
im = axa.imshow(rr ** 2.2, extent=(-lim, lim, -lim, lim), origin="lower", cmap=cm, vmin=0, vmax=1, zorder=0)
clip = Circle((0, 0), lim, transform=axa.transData); im.set_clip_path(clip)
axa.add_patch(Circle((0, 0), lim, fill=False, ec="#B8BCC8", lw=0.7, zorder=0.5))
for g in range(1, int(lim) + 1):
    axa.plot(g * np.cos(t), g * np.sin(t), color="#CDD1DB", lw=0.4, ls=(0, (1, 1.8)), zorder=1)
# stage bands: inter-quartile range of each stage's geodesic distance, then the learned ring
for c, r in enumerate(radii):
    q1, q3 = np.percentile(d[y == c], [25, 75])
    lo, hi = min(q1, r), max(q3, r)                  # IQR of the stage's distances, extended to include its ring
    axa.add_patch(Wedge((0, 0), hi, 0, 360, width=max(hi - lo, 0.04), fc=COLS[c], alpha=0.13, ec="none", zorder=1.5))
    axa.plot(r * np.cos(t), r * np.sin(t), color=COLS[c], lw=1.25, zorder=3,
             path_effects=[pe.withStroke(linewidth=2.4, foreground="white", alpha=0.9)])
# progression axis
ang = np.deg2rad(240)   # pointing to the lower left
axa.annotate("", xy=(lim * 0.97 * np.cos(ang), lim * 0.97 * np.sin(ang)), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color="#595959", lw=0.7), zorder=5)
for c, r in enumerate(radii):                       # stage names at each ring crossing, alternating sides
    cx, cy = r * np.cos(ang), r * np.sin(ang)
    if c % 2 == 0:   # upper-left side of the axis
        axa.text(cx - 0.15, cy + 0.2, NAMES[c], ha="right", va="bottom", fontsize=5.6, fontweight="bold",
                 color=COLS[c], zorder=6, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    else:            # lower-right side of the axis
        axa.text(cx + 0.2, cy - 0.15, NAMES[c], ha="left", va="top", fontsize=5.6, fontweight="bold",
                 color=COLS[c], zorder=6, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
o = np.argsort(-d)
axa.scatter(d[o] * np.cos(theta[o]), d[o] * np.sin(theta[o]), s=5.5, c=[COLS[k] for k in y[o]],
            edgecolors="white", linewidths=0.35, zorder=4)
axa.plot(0, 0, marker="o", ms=2, color="#595959", zorder=6)
axa.set_xlim(-lim * 1.02, lim * 1.02); axa.set_ylim(-lim * 1.02, lim * 1.02)
axa.set_aspect("equal"); axa.axis("off")

# ---------- (b) gap shares
axb.axhline(0.25, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
gcols = COLS[1:]
bp = axb.boxplot([S[:, j] for j in range(4)], widths=0.55, patch_artist=True,
                 medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(lw=0.6, color="#595959"),
                 capprops=dict(lw=0.6, color="#595959"), flierprops=dict(marker=""))
for p, c in zip(bp["boxes"], gcols):
    p.set_facecolor(c); p.set_alpha(0.28); p.set_edgecolor(c); p.set_linewidth(0.9)
rng = np.random.default_rng(0)
for j, c in enumerate(gcols, 1):
    axb.scatter(j + rng.uniform(-0.12, 0.12, len(S)), S[:, j - 1], s=6, color=c,
                edgecolors="white", linewidths=0.3, zorder=3)
m = S.mean(0)
axb.plot(range(1, 5), m, color=ACC, lw=1.1, marker="D", ms=3.2, mec="white", mew=0.5, zorder=4)
axb.set_xticks(range(1, 5))
axb.set_xticklabels([f"{a}\n→{b}" for a, b in zip(NAMES[:-1], NAMES[1:])], fontsize=6.3, linespacing=0.95)
axb.set_ylim(0, 0.45); axb.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axb.tick_params(axis="y", labelsize=6.5)
axb.set_ylabel(r"$\Delta_t/(k_C-k_1)$", labelpad=1.5)
axb.grid(axis="y", color="#EBEBEB", lw=0.5); axb.set_axisbelow(True)
for s in ("top", "right"): axb.spines[s].set_visible(False)

# ---------- labels + shared legend
fig.text(0.01, 0.98, "(a)", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.53, 0.98, "(b)", fontsize=8.5, fontweight="bold", va="top")
h = [Line2D([], [], marker="o", ls="", ms=3.6, mfc=c, mec="white", mew=0.4) for c in COLS]
h.append(Line2D([], [], color=ACC, lw=1.1, marker="D", ms=3, mec="white", mew=0.4))
fig.legend(h, NAMES + ["mean"], loc="lower center", ncol=6, frameon=False, fontsize=6.8,
           handletextpad=0.15, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor="white")
print("mean shares", np.round(m, 3))
