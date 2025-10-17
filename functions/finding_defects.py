# ── standard library ──────────────────────────────────────────────────────────
import re
import time
from pathlib import Path
import math

# ── third‑party ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import gaussian_filter, binary_dilation, label
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from skimage.restoration import denoise_tv_chambolle
import networkx as nx
from typing import Optional , Union , Set      
from numpy.random import default_rng
import pandas as pd
from matplotlib.patches import Rectangle

# ── project‑level ────────────────────────────────────────────────────────────
from . import finetuning_training as utils  

# ─────────────────────────────────────────────────────────────────────────────
# Geometry / search helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_farthest_locations(
    points,
    threshold=None,
    bounds=None,
    grid_size: int = 100,
    margin: float = 0.05,
    plot: bool = True,
    verbose: bool = False,
):
    """Return a boolean *mask* of grid locations farther than *threshold* from
    any point in *points*.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Known atom coordinates.
    threshold : float, optional
        Distance cut‑off; if *None* it is inferred from the mean 1‑NN distance.
    bounds : tuple, optional
        (xmin, xmax, ymin, ymax) to restrict search region; if *None* the
        bounding box of *points* is used.
    grid_size : int
        Resolution of the square search grid.
    margin : float
        Fraction of *bounds* to ignore around the border when suggesting new
        points.
    plot : bool
        Whether to display a diagnostic plot.
    verbose : bool
        Print extra info to *stdout*.

    Returns
    -------
    xs_full, ys_full : 1‑D arrays
        Grid coordinates along X and Y (length = *grid_size*).
    mask : 2‑D bool array, shape (grid_size, grid_size)
        *True* where the distance exceeds *threshold*.
    """
    # Build KD‑tree for fast distance queries
    tree = cKDTree(points)

    # If threshold not given, base it on the average nearest‑neighbour distance
    if threshold is None:
        dists, _ = tree.query(points, k=2)
        threshold = dists[:, 1].mean() * 0.7
    if verbose:
        print(f"threshold = {threshold:.3f}")

    # Define search bounds (optionally expand / contract by *margin*)
    if bounds is None:
        xmin0, xmax0 = points[:, 0].min(), points[:, 0].max()
        ymin0, ymax0 = points[:, 1].min(), points[:, 1].max()
    else:
        xmin0, xmax0, ymin0, ymax0 = bounds
    dx, dy = xmax0 - xmin0, ymax0 - ymin0

    # Create regular sampling grid
    xs_full = np.linspace(xmin0, xmax0, grid_size)
    ys_full = np.linspace(ymin0, ymax0, grid_size)
    xx, yy = np.meshgrid(xs_full, ys_full)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Distance from every grid node to the closest known point
    d_full, _ = tree.query(grid)
    mask = (d_full > threshold).reshape(grid_size, grid_size)

    # Exclude a border of *edge* pixels (margin) so candidates stay interior
    edge = int(grid_size * margin)
    if edge:
        mask[:edge, :] = mask[-edge:, :] = mask[:, :edge] = mask[:, -edge:] = False

    # Dilate mask until it touches an existing point to avoid isolated specks
    point_mask = np.zeros_like(mask)
    ix = np.clip(((points[:, 0] - xmin0) / dx * (grid_size - 1)).astype(int), 0, grid_size - 1)
    iy = np.clip(((points[:, 1] - ymin0) / dy * (grid_size - 1)).astype(int), 0, grid_size - 1)
    point_mask[iy, ix] = True

    curr = mask.copy()
    for _ in range(10):  # max_dilate_steps
        dil = binary_dilation(curr)
        if (dil & point_mask).any():
            break
        curr = dil
    mask = gaussian_filter(curr.astype(float), sigma=1) > 0.6

    # Optional visualisation
    if plot:
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], s=5, c="k", label="points")
        plt.imshow(mask, origin="lower", extent=(xmin0, xmax0, ymin0, ymax0), alpha=0.3)
        plt.title(f"Areas > {threshold:.2f} from nearest point")
        plt.axis("off")
        plt.show()

    return xs_full, ys_full, mask

def iterative_atom_placement(points,
                               threshold=None,
                               bounds=None,
                               grid_size=100,
                               margin=0.05,
                               neighbor_dist=1.3,
                               atom_dist=.9,
                               max_iter=5,
                               plot=True,
                               verbose=False,
                               track=False, seed = None):
    import time
    global_start = time.time()
    if verbose:
        print("\U0001F7E2 START iterative_defect_placement")
    if seed is not None:
        rng = np.random.default_rng(seed)                      
    pts_all = points.copy()
    defects = []
    defects_by_iter = []
    traj = [] if track else None

    if bounds is None:
        xmin, xmax = points[:, 0].min(), points[:, 0].max()
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
    else:
        xmin, xmax, ymin, ymax = bounds

    tree0 = cKDTree(points)
    d2, _ = tree0.query(points, k=2)
    mean_nn = d2[:, 1].mean()

    if plot:
        print("Original Mask :")

    xs, ys, mask = find_farthest_locations(
        pts_all, threshold, (xmin, xmax, ymin, ymax),
        grid_size, margin, plot=plot, verbose=verbose
    )
    labels, n_blob = label(mask)
    if n_blob == 0:
        if verbose:
            print("Stopped at iter 0: no blobs")
        return defects, np.empty((0, 2))

    large_ids = [i for i in range(1, n_blob + 1) if (labels == i).sum() > mean_nn * 5]
    if verbose:
        print(f"Found {len(large_ids)} large blobs")

    missing_large = []

    if len(large_ids) > 0:
        result = estimate_hex_lattice(points)
        if result is False:
            missing_large = np.empty((0, 2))
        else:
            cell, origin, _ = result
            all_finals = []
            all_origins = []
            all_lids = []

            inv_cell = np.linalg.inv(cell)
            for lid in large_ids:
                coords = np.argwhere(labels == lid)
                xsb = xs[coords[:, 1]]
                ysb = ys[coords[:, 0]]
                pts_cart = np.column_stack((xsb, ysb))
                blob_center = pts_cart.mean(axis=0)
                _, idx = tree0.query(blob_center)
                origin_blob = points[idx]

                pad = 10
                xminb, xmaxb = xsb.min() - pad, xsb.max() + pad
                yminb, ymaxb = ysb.min() - pad, ysb.max() + pad

                pts_blob = pts_all[
                    (pts_all[:, 0] >= xminb) & (pts_all[:, 0] <= xmaxb) &
                    (pts_all[:, 1] >= yminb) & (pts_all[:, 1] <= ymaxb)]
                if pts_blob.size == 0:
                    continue

                frac = (pts_blob - origin_blob) @ inv_cell
                grid_blob = np.round(frac).astype(int)
                occ = set(map(tuple, grid_blob))

                fmin, fmax = frac.min(axis=0), frac.max(axis=0)
                all_sites = [(i, j) for i in range(int(np.floor(fmin[0])), int(np.ceil(fmax[0])) + 1)
                            for j in range(int(np.floor(fmin[1])), int(np.ceil(fmax[1])) + 1)]
                empty = np.array([s for s in all_sites if s not in occ])
                if empty.size == 0:
                    continue
                cart_empty = empty @ cell + origin_blob

                ix_e = np.clip(((cart_empty[:, 0] - xmin) / (xmax - xmin) * (grid_size - 1)).astype(int), 0, grid_size - 1)
                iy_e = np.clip(((cart_empty[:, 1] - ymin) / (ymax - ymin) * (grid_size - 1)).astype(int), 0, grid_size - 1)
                in_blob = labels[iy_e, ix_e] == lid
                final = cart_empty[in_blob]
                if final.size == 0:
                    continue

                missing_large.append(final)
                all_finals.append(final)
                all_origins.append(origin_blob)
                all_lids.append(lid)

            if plot and all_finals:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(pts_all[:, 0], pts_all[:, 1], s=1)
                for final, origin, lid in zip(all_finals, all_origins, all_lids):
                    ax.imshow(labels == lid, origin='lower', extent=(xmin, xmax, ymin, ymax), alpha=0.1, cmap='Reds')
                    ax.scatter(final[:, 0], final[:, 1], s=10, c='r')
                    ax.scatter(origin[0], origin[1], marker='*', s=100)
                ax.set_title("All Large Blobs (Stars = Center for Linear Transform)")
                ax.axis('off')
                plt.show()

    missing_large = np.vstack(missing_large) if len(missing_large) > 0 else np.empty((0, 2))
    pts_all = np.vstack((points, missing_large))

    for it in range(max_iter):
        iter_start = time.time()
        if verbose:
            print(f"\n--- Iteration {it+1}/{max_iter} ---")

        plot2 = plot and it == 0
        xs, ys, mask = find_farthest_locations(
            pts_all, threshold, (xmin, xmax, ymin, ymax),
            grid_size, margin, plot=plot2, verbose=verbose
        )
        labels, n_blob = label(mask)

        new_pts = []
        if verbose:
            print(f"n_blob = {n_blob}, mask sum = {mask.sum()}")
        for b in range(1, n_blob + 1):
            coords = np.argwhere(labels == b)
            far_thr = mean_nn * atom_dist
            close_thr = mean_nn * neighbor_dist
            if verbose:
                print(f"Blob {b}: {len(coords)} candidate points")

            valid = []
            for y, x in coords:
                cand = np.array([xs[x], ys[y]])
                d_at = np.linalg.norm(points - cand, axis=1)
                at_count = (d_at < close_thr).sum()

                if len(defects) > 0:
                    d_df = np.linalg.norm(np.vstack(defects) - cand, axis=1)
                    df_count = (d_df < close_thr).sum()
                else:
                    df_count = 0

                if at_count + df_count >= 3:
                    valid.append((y, x))

            if not valid:
                continue
            if seed is not None:
                y0, x0 = valid[rng.integers(len(valid))]   
            else:
                y0, x0 = valid[np.random.randint(len(valid))]

            cand = np.array([xs[x0], ys[y0]])
            step = 0.5

            cand_start = time.time()
            if verbose:
                print(f"\u2192 Starting candidate descent for Blob {b}, point ({xs[x0]:.1f}, {ys[y0]:.1f})")
            for _ in range(50):
                occ_list = [points]
                if len(missing_large) > 0:
                    occ_list.append(missing_large)
                if len(defects) > 0:
                    occ_list.append(np.vstack(defects))
                occ = np.vstack(occ_list)

                d_all = np.linalg.norm(occ - cand, axis=1)

                d_at = np.linalg.norm(points - cand, axis=1)
                d_ml = np.linalg.norm(missing_large - cand, axis=1) if missing_large.size else np.array([])
                d_df = np.linalg.norm(np.vstack(defects) - cand, axis=1) if defects else np.array([])

                close_count = (d_at < close_thr).sum() + (d_ml < close_thr).sum() + (d_df < close_thr).sum()
                if d_all.min() > far_thr and close_count >= 3:
                    break
                if _ % 10 == 0 and verbose:
                    print(f"   step {_}: d_all.min()={d_all.min():.2f}, far_thr={far_thr:.2f}, close_count={close_count}")

                i0 = d_all.argmin()
                v_away = cand - occ[i0]
                v_away /= np.linalg.norm(v_away)
                if close_count < 3:
                    ia = np.argsort(d_at)[:3]
                    v = points[ia].mean(axis=0) - cand
                    v /= np.linalg.norm(v)
                    direction = v_away + v
                else:
                    direction = v_away
                direction /= np.linalg.norm(direction)
                cand += step * direction
            else:
                continue

            if verbose:
                print(f"\u2705 Candidate placed after {_+1} steps in {time.time() - cand_start:.2f}s")
            new_pts.append(cand)

        if not new_pts:
            if verbose:
                print(f"No new defects at iter {it}")
            break

        defects.extend(new_pts)
        defects_by_iter.append(np.array(new_pts))

        all_components = [points]
        if missing_large.size:
            all_components.append(missing_large)
        if len(defects) > 0:
            all_components.append(np.vstack(defects))
        pts_all = np.vstack(all_components)

        if verbose:
            print(f"\u23F1\uFE0F Iteration {it+1} finished in {time.time() - iter_start:.2f}s")
            print(f"Iteration {it+1}: found {len(new_pts)} defects")

    if plot:
        print("Final Dots located on Final Mask")
        xs2, ys2, fm = find_farthest_locations(
            pts_all, threshold, (xmin, xmax, ymin, ymax),
            grid_size, margin, plot=False, verbose=False
        )
        cmap = plt.cm.jet(np.linspace(0, 1, len(defects_by_iter)))
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], c='k', s=1)
        for i, pts in enumerate(defects_by_iter):
            plt.scatter(pts[:, 0], pts[:, 1], s=20, color=cmap[i], label=f'iter {i+1}')
        if track and traj:
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], 'w--', lw=1.5)
        plt.imshow(fm, origin='lower', extent=(xmin, xmax, ymin, ymax), alpha=0.3)
        if missing_large.size:
            plt.scatter(missing_large[:, 0], missing_large[:, 1], c='r', s=20, label="Large Blob Dots")
        plt.legend(loc='upper right', fontsize=5)
        plt.axis('off')
        plt.show()

    if verbose:
        print("\U0001F7E3 END iterative_defect_placement — total time:", round(time.time() - global_start, 2), "seconds")
    defects = np.vstack(defects) if defects else np.empty((0, 2))
    return defects, missing_large

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_tif_folder(folder_path: Union[str, Path]):
    """Load *all* ``.tif`` files in *folder_path* and return a list of PIL images
    **and** their absolute paths."""
    folder_path = Path(folder_path)
    files = [f for f in folder_path.iterdir() if f.suffix.lower() == ".tif"]
    return [Image.open(f) for f in files], list(map(str, files))


def mx_val(fname: str):
    """Extract the *Mx* numeric value encoded in an image filename.

    Examples
    --------
    >>> mx_val("sample_2.5_Mx_image.tif")
    2.5
    """
    m = re.search(r"(?:^|[_\s-])(\d+(?:\.\d+)?)_?Mx(?:_|$)", fname, re.I)
    return float(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Image loading / scaling helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_images_with_pixel_sizes(
    folder: Path,
    ps_by_mx: dict[float, float],
    resize_px: int,
    skip_idx: Optional[Set[int]] = None,
    eps: float = 1e-3,
):
    """Load ‑> square‑crop ‑> resize a folder of *tif* images while preserving
    pixel‑size metadata.

    Returns
    -------
    images_resized : list[np.ndarray]
    px_sizes_orig  : list[float]  – nm / px in the raw image
    px_sizes_new   : list[float]  – nm / px after the crop/resample
    """
    skip_idx = skip_idx or set()
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() == ".tif" and f.is_file()])
    files = [f for i, f in enumerate(files) if i not in skip_idx]

    images_resized, px_sizes_orig, px_sizes_new = [], [], []
    for f in files:
        mx = mx_val(f.name)
        if mx is None:
            raise ValueError(f"Cannot extract Mx from '{f.name}'")
        try:
            mx_key = next(k for k in ps_by_mx if abs(k - mx) < eps)
        except StopIteration:
            raise ValueError(f"Mx={mx} in '{f.name}' not in ps_by_mx")
        orig_nm_per_px = ps_by_mx[mx_key]

        im = Image.open(f)
        w, h = im.size
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        im = im.resize((resize_px, resize_px), Image.LANCZOS)

        scale = side / resize_px
        images_resized.append(np.asarray(im))
        px_sizes_orig.append(orig_nm_per_px)
        px_sizes_new.append(orig_nm_per_px * scale)

    return images_resized, px_sizes_orig, px_sizes_new


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def image_preprocess(img, denoise = False, resize_px = 512, sigma: float = 1.5,weight: float = 0.001):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    if denoise:
        im = Image.fromarray(denoise_tv_chambolle(gaussian_filter(im, sigma=sigma), weight=weight))
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    im = im.resize((resize_px, resize_px), Image.LANCZOS)
    arr = np.asarray(im)
    return arr



def plot_grid(
    images,
    *,
    model=None,
    mask_func=None,
    show_mask: bool = False,
    sigma: float = 1.5,
    weight: float = 0.001,
    resize_px: int = 512,
    cmap: str = "gray",
    title_prefix: str = "Image",   # set to None for no titles
    figsize_factor: int = 3,
    thresh: bool = True,
    low_thresh: float = 0.3,
    high_thresh: float = 0.65,
    min_pixels: int = 400,
    denoise: bool = False,
    iter: int = 1,
    cols: int = None,
    pixel_sizes_nm_per_px=None,
    scale_bar_nm: float = 1.0,
    scale_bar_height_px: int = 4,
    scale_bar_margin_px: int = 8,
    scale_bar_color: str = "white",
    scale_bar_edge_color: str = "black",
    show_scale_label: bool = False,
    label_fontsize: int = 9,
    overlay_df=None,
    overlay_meta=None,
    overlay_atoms=None,
    overlay_defects=None,
    atom_size=8,
    defect_size=28,
    atom_color="tab:blue",
    defect_color="red",
    group_colors=None,
):
    """Quick n×n image mosaic with optional segmentation overlays, scale bars, and XY points."""

    def _norm_percent(p):
        if p is None:
            return None
        s = str(p).strip()
        return s.replace(".", "_")

    n = len(images)
    if cols is None:
        cols = math.ceil(n ** 0.5)
        rows = math.ceil(n / cols)
    else:
        rows = math.ceil(n / cols)

    if pixel_sizes_nm_per_px is None:
        px_sizes = [None] * n
    elif isinstance(pixel_sizes_nm_per_px, (int, float)):
        px_sizes = [float(pixel_sizes_nm_per_px)] * n
    else:
        if len(pixel_sizes_nm_per_px) != n:
            raise ValueError("Length of pixel_sizes_nm_per_px must match number of images.")
        px_sizes = [float(v) if v is not None else None for v in pixel_sizes_nm_per_px]

    use_df = overlay_df is not None and len(overlay_df) > 0
    if use_df:
        df = overlay_df.copy()
        if "percent" in df.columns:
            df["percent"] = df["percent"].astype(str).str.replace(".", "_", regex=False)
        if "sample_id" in df.columns:
            df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype("Int64")
        if "group" in df.columns:
            df["group"] = pd.to_numeric(df["group"], errors="coerce").fillna(0).astype(int)
        if "label" in df.columns:
            df["label"] = df["label"].astype(str)
        for col in ("x", "y"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if use_df and overlay_meta is None:
        raise ValueError("overlay_meta (list of {'percent','sample_id'}) is required when overlay_df is provided.")
    if overlay_meta is not None and len(overlay_meta) != n:
        raise ValueError("overlay_meta length must match number of images.")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_factor, rows * figsize_factor))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, img, nm_per_px) in enumerate(zip(axes, images, px_sizes)):
        im = Image.fromarray(img) if isinstance(img, np.ndarray) else img

        if denoise:
            arr_d = denoise_tv_chambolle(gaussian_filter(np.asarray(im), sigma=sigma), weight=weight)
            im = Image.fromarray(arr_d.astype(np.asarray(im).dtype))

        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        right = left + side
        bottom = top + side
        im = im.crop((left, top, right, bottom))

        im = im.resize((resize_px, resize_px), Image.LANCZOS)
        scale_factor = resize_px / float(side)
        arr = np.asarray(im)

        if show_mask:
            if mask_func is not None:
                ax.imshow(mask_func(im), cmap=cmap)
            elif model is not None:
                prob = arr
                for _ in range(iter):
                    prob, _ = model.predict(prob.squeeze(), thresh=0.5)
                prob = prob.squeeze()
                if thresh:
                    low_bin = prob >= low_thresh
                    labels_x, _ = ndi.label(low_bin)
                    sizes = np.bincount(labels_x.ravel())
                    keep = sizes >= min_pixels
                    keep[0] = False
                    large = keep[labels_x]
                    small = prob >= high_thresh
                    prob = np.logical_or(large, small).astype(np.uint8)
                ax.imshow(prob, cmap=cmap)
            else:
                ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(arr, cmap=cmap)

        ax.axis("off")
        if title_prefix is not None:
            ax.set_title(f"{title_prefix} {i+1}", fontsize=10)

        if nm_per_px is not None and nm_per_px > 0:
            bar_len_px_orig = scale_bar_nm / nm_per_px
            bar_len_px = max(1, min(bar_len_px_orig * scale_factor, resize_px - 2 * scale_bar_margin_px))
            x0 = resize_px - scale_bar_margin_px - bar_len_px
            y0 = resize_px - scale_bar_margin_px - scale_bar_height_px
            ax.add_patch(Rectangle((x0, y0), bar_len_px, scale_bar_height_px,
                                   facecolor=scale_bar_color, edgecolor=scale_bar_edge_color,
                                   linewidth=0.8, zorder=5))
            if show_scale_label:
                ax.text(x0 + bar_len_px / 2, y0 - 4, f"{scale_bar_nm:g} nm",
                        ha="center", va="bottom", fontsize=label_fontsize,
                        color=scale_bar_color, zorder=6)

        if use_df:
            meta = overlay_meta[i]
            p = _norm_percent(meta.get("percent"))
            sid = int(meta.get("sample_id"))
            sub = df[(df["percent"] == p) & (df["sample_id"] == sid)]
            if len(sub):
                sub_a = sub[sub["label"] == "atom"]
                if len(sub_a):
                    xs = (sub_a["x"].to_numpy() - left) * scale_factor
                    ys = (sub_a["y"].to_numpy() - top) * scale_factor
                    m = (xs >= 0) & (xs < resize_px) & (ys >= 0) & (ys < resize_px)
                    if np.any(m):
                        if group_colors is not None and "group" in sub_a:
                            cols = [group_colors.get(int(g), atom_color) for g in sub_a["group"].to_numpy()]
                            ax.scatter(xs[m], ys[m], s=atom_size, c=np.array(cols)[m], marker='o', linewidths=0)
                        else:
                            ax.scatter(xs[m], ys[m], s=atom_size, c=atom_color, marker='o', linewidths=0)

                sub_d = sub[sub["label"] == "defect"]
                if len(sub_d):
                    xs = (sub_d["x"].to_numpy() - left) * scale_factor
                    ys = (sub_d["y"].to_numpy() - top) * scale_factor
                    m = (xs >= 0) & (xs < resize_px) & (ys >= 0) & (ys < resize_px)
                    if np.any(m):
                        if group_colors is not None and "group" in sub_d:
                            cols = [group_colors.get(int(g), defect_color) for g in sub_d["group"].to_numpy()]
                            ax.scatter(xs[m], ys[m], s=defect_size, c=np.array(cols)[m], marker='o', linewidths=0)
                        else:
                            ax.scatter(xs[m], ys[m], s=defect_size, c=defect_color, marker='o', linewidths=0)

        if overlay_atoms is not None and not use_df:
            pts = overlay_atoms[i] if isinstance(overlay_atoms, (list, tuple)) else overlay_atoms
            if pts is not None and len(pts):
                pts = np.asarray(pts, dtype=float)
                xs = (pts[:, 0] - left) * scale_factor
                ys = (pts[:, 1] - top) * scale_factor
                m = (xs >= 0) & (xs < resize_px) & (ys >= 0) & (ys < resize_px)
                if np.any(m):
                    ax.scatter(xs[m], ys[m], s=atom_size, c=atom_color, marker='o', linewidths=0)

        if overlay_defects is not None and not use_df:
            pts = overlay_defects[i] if isinstance(overlay_defects, (list, tuple)) else overlay_defects
            if pts is not None and len(pts):
                pts = np.asarray(pts, dtype=float)
                xs = (pts[:, 0] - left) * scale_factor
                ys = (pts[:, 1] - top) * scale_factor
                m = (xs >= 0) & (xs < resize_px) & (ys >= 0) & (ys < resize_px)
                if np.any(m):
                    ax.scatter(xs[m], ys[m], s=defect_size, c=defect_color, marker='o', linewidths=0)

    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()

def sigmoid(x: Union[np.ndarray, float]):
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# High‑level analysis / plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_point_masks_grid(
    images,
    *,
    ensemble_pred,
    defect_model,
    sigma: float = 1.5,
    weight: float = 0.001,
    t_values=None,
    low_thresh: float = 0.30,
    high_thresh: float = 0.65,
    min_pixels: int = 400,
    dilate_iter: int = 3,
    figsize_factor: int = 3,
    title_prefix: str = "Img",
    plot_mask_grid: bool = False,
    rows = None,
    plot_thresholds: bool = False
):
    t_values = t_values or [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    n = len(images)
    if rows is None:
        cols = math.ceil(n ** 0.5)
        rows = math.ceil(n / cols)
    else:
        cols = math.ceil(n / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_factor, rows * figsize_factor))
    axes = np.array(axes, ndmin=2).reshape(rows, cols)

    img_stats, masks = [], []
    for idx, (ax, image) in enumerate(zip(axes.ravel(), images)):
        img_arr, _, pts_all, _ = utils.get_total_points_ensemble(
            ensemble_pred, image, plot=False, plot2=False, normalize=True, t_values=t_values
        )

        missing_small, missing_large = iterative_atom_placement(pts_all, grid_size=200, plot=False, seed=42)
        missing_small = np.asarray(missing_small).reshape(-1, 2)
        missing_large = np.asarray(missing_large).reshape(-1, 2)
        cands = [a for a in (missing_small, missing_large) if a.size]
        missing = np.vstack(cands) if cands else np.empty((0, 2))
        pts_all = np.vstack((pts_all, missing)) if missing.size else pts_all

        im_filt = Image.fromarray(denoise_tv_chambolle(gaussian_filter(image, sigma=sigma), weight=weight))
        mask_prob = utils.get_mask(defect_model, im_filt)

        low_bin = mask_prob >= low_thresh
        labels, _ = ndi.label(low_bin)
        keep = np.bincount(labels.ravel()) >= min_pixels
        keep[0] = False
        large = keep[labels]
        small = mask_prob >= high_thresh
        mask = np.logical_or(large, small).astype(np.uint8)
        mask_dil = binary_dilation(mask, iterations=dilate_iter) if dilate_iter else mask
        masks.append(mask_dil)

        if plot_thresholds:
            fig_th, axs = plt.subplots(1, 5, figsize=(12.5, 2.8), constrained_layout=True)
            axs[0].imshow(mask_prob, cmap="magma"); axs[0].set_title("prob")
            axs[1].imshow(low_bin, cmap="gray"); axs[1].set_title(f">= low {low_thresh:.2f}")
            axs[2].imshow(large, cmap="gray"); axs[2].set_title("size‑filtered (large)")
            axs[3].imshow(small, cmap="gray"); axs[3].set_title(f">= high {high_thresh:.2f}")
            axs[4].imshow(mask_dil, cmap="gray"); axs[4].set_title("final mask")
            for a in axs: a.axis("off")

        h, w = mask_dil.shape
        ix = pts_all[:, 0].astype(int).clip(0, w - 1)
        iy = pts_all[:, 1].astype(int).clip(0, h - 1)
        defects = pts_all[mask_dil[iy, ix]]
        regular = pts_all[~mask_dil[iy, ix]]
        _, col, counts = cluster_by_size(regular, defects, img_arr, plot=False)

        ax.imshow(img_arr, cmap="gray")
        if regular.size:
            ax.scatter(regular[:, 0], regular[:, 1], c="cornflowerblue", s=1)
        if defects.size:
            ax.scatter(defects[:, 0], defects[:, 1], color=col, s=10)
        ax.set_title(f"{title_prefix} Image {idx+1}", fontsize=9)
        ax.axis("off")

        img_stats.append({
            "image_index": idx,
            "regular_points": regular,
            "defect_points": defects,
            "colors": col,
            "cluster_counts": counts,
            "mask": mask_dil,
        })

    extra = axes.size - n
    if extra > 0:
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)

    plt.rcParams['axes.facecolor'] = 'black'
    plt.tight_layout()

    if plot_mask_grid:
        fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * figsize_factor, rows * figsize_factor))
        axes2 = np.array(axes2, ndmin=2).reshape(rows, cols)
        for j, (ax, m) in enumerate(zip(axes2.ravel(), masks)):
            ax.imshow(m, cmap="gray")
            ax.set_title(f"{title_prefix} Defect Mask {j+1}", fontsize=9)
            ax.axis("off")
        extra2 = axes2.size - len(masks)
        if extra2 > 0:
            for ax in axes2.ravel()[len(masks):]:
                ax.set_visible(False)
        plt.tight_layout()

    return img_stats

def vacancy_percent(sample):
    vac = sum(len(s['defect_points']) for s in sample)
    reg = sum(len(s['regular_points']) for s in sample)
    return 100 * vac / (vac + reg)


def bootstrapping_plot(loaded_data, sample_sizes=[1, 5, 10, 15, 20, 25], B=4000, font_size=None):
    results_all = []

    for label, data in loaded_data.items():
        image_stats = data["img_stats"]
        N_imgs = len(image_stats)
        sample_sizes_full = sample_sizes + [N_imgs]
        rng = default_rng(42)

        records = []
        for n in sample_sizes_full:
            boots = [
                vacancy_percent([image_stats[i] for i in rng.choice(N_imgs, size=n, replace=True)])
                for _ in range(B)
            ]
            boots = np.asarray(boots)
            records.append({
                'size': n,
                'mean': boots.mean(),
                'se': boots.std(ddof=1),
                'ci_lower': np.percentile(boots, 2.5),
                'ci_upper': np.percentile(boots, 97.5),
            })

        results = pd.DataFrame(records).sort_values('size')
        se_full = results.loc[results['size'] == N_imgs, 'se'].iloc[0]
        results['rel_se'] = results['se'] / se_full
        results['ci_minus'] = results['mean'] - results['ci_lower']
        results['ci_plus'] = results['ci_upper'] - results['mean']
        results['label'] = label.replace('_', '.')

        results_all.append(results)

    df_all = pd.concat(results_all, ignore_index=True)

    if font_size is not None:
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': int(font_size * 1.1),
            'xtick.labelsize': int(font_size * 0.9),
            'ytick.labelsize': int(font_size * 0.9),
            'legend.fontsize': int(font_size * 0.9),
        })

    plt.figure(figsize=(8, 6))
    colors = {'5': 'tab:red', '9.1': 'tab:blue', '12.5': 'tab:green'}

    for lbl, group in df_all.groupby('label'):
        plt.errorbar(
            group['size'],
            group['mean'],
            yerr=[group['ci_minus'], group['ci_plus']],
            fmt='o-',
            capsize=4,
            linewidth=2,
            markersize=6,
            label=f'{lbl}% HF',
            color=colors.get(lbl, None)
        )

    plt.xlabel('# images collected')
    plt.ylabel('Vacancy percentage (%)')
    plt.title('Bootstrapped 95% CI Across HF Concentrations')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results_all

def make_sample_csvs(loaded_data, out_dir="."):
    for sample, data in loaded_data.items():
        rows = []
        for s in data["img_stats"]:
            vac = len(s["defect_points"])
            reg = len(s["regular_points"])
            rows.append(
                {
                    "image_index": s["image_index"],
                    'sample': sample,
                    "num_atoms":  reg + vac,
                    "num_vacancies": vac,
                    "percent_vacancy": 100 * vac / (reg + vac) if reg + vac else 0,
                    "file_name":  s["file_name"]
                }
            )
        pd.DataFrame(rows).to_csv(f"{out_dir}/{sample}_stats.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Lattice estimation & clustering
# ─────────────────────────────────────────────────────────────────────────────

def estimate_hex_lattice(points, img=None, margin: int = 15, n_clusters: int = 3):
    """Estimate basis vectors of a hexagonal lattice from point cloud."""
    if img is not None:
        h, w = img.shape
        mask = (points[:, 0] > margin) & (points[:, 0] < w - margin) & (points[:, 1] > margin) & (points[:, 1] < h - margin)
    else:
        mask = np.ones(len(points), bool)
    tri = Delaunay(points)
    sims = tri.simplices[np.all(mask[tri.simplices], axis=1)] if img is not None else tri.simplices.copy()

    edges = np.unique(np.sort(np.vstack([sims[:, [0, 1]], sims[:, [1, 2]], sims[:, [2, 0]]]), axis=1), axis=0)
    vecs = points[edges[:, 1]] - points[edges[:, 0]]
    length = np.median(np.linalg.norm(vecs, axis=1))

    ang = np.mod(np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])), 180)
    hist, bin_edges = np.histogram(ang, bins=360, range=(0, 180))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peaks_idx, props = find_peaks(hist, distance=30, height=20)
    if len(peaks_idx) < n_clusters:
        return False
    top3_idx = peaks_idx[np.argsort(props["peak_heights"])[-3:]]
    peaks = np.sort(bin_centers[top3_idx])

    # closest pair ≈ 60° apart
    best_pair, min_diff = None, np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            diff = abs((peaks[i] - peaks[j] + 180) % 180 - 60)
            if diff < min_diff:
                min_diff, best_pair = diff, (peaks[i], peaks[j])
    theta1, theta2 = map(np.deg2rad, best_pair)
    cell = np.vstack([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]) * length
    return cell, points.mean(axis=0), length


def cluster_by_size(points, defects, img_arr, plot: bool = True):
    """Group vacancy points into clusters sized 1, 2, 3, or 4+ atoms."""
    total_cluster_counts = {}
    coords = np.vstack([points, defects])
    labels = np.hstack([np.zeros(len(points)), np.ones(len(defects))])

    tri = Delaunay(coords)
    edges = {tuple(sorted([s[i], s[(i + 1) % 3]])) for s in tri.simplices for i in range(3)}
    valid = [e for e in edges if labels[e[0]] == 1 and labels[e[1]] == 1]

    g = nx.Graph()
    g.add_edges_from(valid)
    components = list(nx.connected_components(g))

    sizes = np.zeros(len(coords), dtype=int)
    for comp in components:
        for node in comp:
            sizes[node] = len(comp)
    sizes[(labels == 1) & (sizes == 0)] = 1

    # count clusters
    if components:
        num_singles = np.sum((labels == 1) & (sizes == 1) & (~np.isin(np.arange(len(sizes)), np.concatenate([list(c) for c in components]))))
        if num_singles:
            total_cluster_counts[1] = total_cluster_counts.get(1, 0) + num_singles
    for comp in components:
        sz_group = 4 if len(comp) > 3 else len(comp)
        total_cluster_counts[sz_group] = total_cluster_counts.get(sz_group, 0) + 1

    cluster_colors = ["orange", "#4daf4a", "#f781bf", "#e41a1c"]
    vacancy_colors = [cluster_colors[(4 if s > 3 else s) - 1] for s in sizes[len(points):]]

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img_arr, cmap="gray", origin="lower")
        ax1.scatter(defects[:, 0], defects[:, 1], s=25, c=vacancy_colors)
        ax1.scatter(points[:, 0], points[:, 1], c="cornflowerblue", s=0.5)
        ax1.axis("off")

        u_sizes = sorted(total_cluster_counts.keys())
        counts = [total_cluster_counts[s] for s in u_sizes]
        for sz, ct in zip(u_sizes, counts):
            color = "#e41a1c" if sz == 4 else cluster_colors[sz - 1]
            ax2.bar(sz, ct, color=color, edgecolor=color, width=0.9)
            ax2.text(sz, ct, str(ct), ha="center", va="bottom", fontsize=20)
        ax2.set_xticks(u_sizes)
        ax2.set_xticklabels(["4+" if s == 4 else str(s) for s in u_sizes])
        ax2.set_xlabel("Vacancy Cluster Size")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.set_yticks([])
        plt.tight_layout()

    return sizes[len(points):], vacancy_colors, total_cluster_counts
