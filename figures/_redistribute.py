"""Angle redistribution for the disk plot (copied verbatim from plot_results.py)."""
import numpy as np


def redistribute_angles_per_class(theta_arr, d_H_arr, label_arr, radii,
                                  num_classes, mode='stretch', min_frac=0.35):
    """Spread same-class samples around their ring (on a copy).

    Both modes keep each class's angular ORDER and anchor the class on the
    sample whose d_H is closest to its learned ring radius r_c, so the ring is
    not rotated relative to the projection.

    mode='uniform'  every adjacent pair exactly 2π/n_c apart. Simple and never
                    overlaps, but it discards the spacing entirely: a tight
                    doublet and a lone outlier come out looking identical.

    mode='stretch'  keep the spacings in PROPORTION to the projection's, and
                    magnify them to fill the ring. Points in a 2-D projection
                    of one class usually occupy a narrow arc with one large
                    empty gap; plotting those raw leaves the class as an
                    unreadable blob. So:

                      1. cut the circle at the class's LARGEST angular gap —
                         that gap is empty space, not information;
                      2. the remaining n_c-1 spacings become a line segment of
                         length L, which is rescaled to 2π minus one uniform
                         slot kept as a visible seam;
                      3. a floor of min_frac × the uniform spacing is reserved
                         for every pair first, and only the surplus is shared
                         out in proportion to the original gaps.

                    Step 3 is what guarantees no overlap. min_frac trades the
                    two readings off continuously: 0 is pure proportional
                    stretch (a near-coincident pair stays near-coincident and
                    will overplot), 1 collapses to mode='uniform'. The default
                    0.35 keeps relative spacing legible while holding adjacent
                    points visually apart.

    Either way the angle is still not a metric quantity — it is a rendering of
    a 2-D projection's ordering. Only the radius is trained.
    """
    theta_out = theta_arr.copy().astype(float)
    if mode not in ('uniform', 'stretch'):
        raise ValueError(f"unknown redistribute mode {mode!r}")

    for ci in range(num_classes):
        if ci >= len(radii):
            continue
        idx = np.where(label_arr == ci)[0]
        nc = len(idx)
        if nc < 2:
            continue
        anchor_local = int(np.argmin(np.abs(d_H_arr[idx] - radii[ci])))
        anchor_theta = float(theta_arr[idx[anchor_local]])
        th = np.asarray(theta_arr[idx], dtype=float)
        sort_order = np.argsort(th)
        anchor_rank = int(np.where(sort_order == anchor_local)[0][0])

        if mode == 'uniform':
            offs = (np.arange(nc) - anchor_rank) * (2.0 * np.pi / nc)
        else:
            th_sorted = th[sort_order]
            # n_c circular gaps; the last wraps past 2π back to the first.
            gaps = np.diff(np.concatenate([th_sorted,
                                           [th_sorted[0] + 2.0 * np.pi]]))
            seam = int(np.argmax(gaps))
            # Re-cut so the widest gap sits between the last and first point.
            roll = (seam + 1) % nc
            order = np.roll(sort_order, -roll)
            inner = np.roll(gaps, -roll)[:nc - 1]      # drop the seam itself

            slot = 2.0 * np.pi / nc                    # one uniform slot
            target = 2.0 * np.pi - slot                # room for the n_c-1 gaps
            floor = min_frac * target / (nc - 1)
            surplus = max(target - (nc - 1) * floor, 0.0)
            tot = float(inner.sum())
            share = (inner / tot) if tot > 0 else np.full(nc - 1, 1.0 / (nc - 1))
            new_inner = floor + surplus * share

            pos = np.concatenate([[0.0], np.cumsum(new_inner)])
            # anchor_rank indexes the ascending order; find it in the rolled one
            a = int(np.where(order == anchor_local)[0][0])
            offs = pos - pos[a]
            sort_order = order

        for k, sort_idx in enumerate(sort_order):
            theta_out[idx[sort_idx]] = anchor_theta + offs[k]
    return theta_out
