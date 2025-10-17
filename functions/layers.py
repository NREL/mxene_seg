import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import pandas as pd
from itertools import permutations
from sklearn.cluster import DBSCAN
from skimage import measure
from matplotlib.patches import Polygon
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import cKDTree
import itertools as it
from matplotlib.gridspec import GridSpec
from scipy.stats import mode
from skimage import draw





def make_perfect_dots(dots, defects, basis, image_arr, adatoms=None, plot=False, test = False, verbose = False):
    basis, err = refine_basis(
        np.vstack([dots, defects]), basis,
        dθ=0.2, spanθ=4.0,
        ds=0.01, span_s=0.1
    )


    # Then coarse refinement
    basis, err = refine_basis(
        np.vstack([dots, defects]), basis,
        dθ=0.05, spanθ=2.0,
        ds=0.001, span_s=0.05
    )


    #if verbose:
        #print(f"Error in basis refinement: {err:.4f}")
    snapped, lbl = snap_in_place(dots, defects, basis)
    perfect_dots     = snapped[lbl == 0]
    perfect_defects  = snapped[lbl == 1]

    #if adatoms is not None:
        #perfect_adatoms = snap_in_place(adatoms, basis)


    all_snapped = np.vstack([perfect_dots, perfect_defects])
    #if adatoms is not None:
        #all_snapped = np.vstack([all_snapped, perfect_adatoms])


    frac_all = np.dot(all_snapped, np.linalg.inv(basis))
    int_coords = np.round(frac_all).astype(int)
    hull = ConvexHull(int_coords)
    hull_path = Delaunay(int_coords[hull.vertices])

    min_x, min_y = int_coords.min(axis=0)
    max_x, max_y = int_coords.max(axis=0)
    grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x+1), np.arange(min_y, max_y+1))
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    in_hull = hull_path.find_simplex(grid_points) >= 0
    missing = grid_points[in_hull & ~np.isin(grid_points.view([('', grid_points.dtype)] * 2), 
                                             int_coords.view([('', int_coords.dtype)] * 2)).ravel()]

    if missing.size > 0:
        h, w = image_arr.shape[:2]
        added_atoms = np.dot(missing, basis)
        inside_image = (
            (added_atoms[:, 0] > 10) & (added_atoms[:, 0] < w - 10) &
            (added_atoms[:, 1] > 10) & (added_atoms[:, 1] < h - 10)
        )
        missing = missing[inside_image]
        added_atoms = added_atoms[inside_image]
        #if verbose:
            #print(f"Found {len(added_atoms)} missing atoms within image bounds.")
 
        h, w = image_arr.shape[:2]
        new_atoms   = added_atoms
        perfect_dots    = np.vstack([perfect_dots, new_atoms])
    else:
        new_defects = np.empty((0, 2))
        new_atoms   = np.empty((0, 2))

    b1 = np.dot([0, 1], basis)
    b2 = np.dot([1, 0], basis)
    rotation_matrix = [[np.cos(-np.pi / 6), -np.sin(-np.pi / 6)],
                       [np.sin(-np.pi / 6),  np.cos(-np.pi / 6)]]
    b1_rot = np.dot(b1, rotation_matrix)
    b2_rot = np.dot(b2, rotation_matrix)

    center = calculate_center(dots)
    
    if added_atoms.size:
        margin = 30                             
        h, w = image_arr.shape[:2]
        d = np.min(np.c_[added_atoms[:, 0],
                        w - 1 - added_atoms[:, 0],
                        added_atoms[:, 1],
                        h - 1 - added_atoms[:, 1]], axis=1)

        if verbose:
            print(f"added {len(added_atoms)} atoms")
            if np.any(d > margin):
                print("WARNING: added atom(s) lie near the image center")
        if test and np.any(d > margin):
            return np.sum(d > margin)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_arr, cmap="gray", origin="lower")

        # original snapped atoms
        orig_atoms = perfect_dots[:-len(added_atoms)] if added_atoms.size else perfect_dots
        plt.scatter(orig_atoms[:, 0], orig_atoms[:, 1], s=5, c="b", label="atoms")

        # newly added atoms
        if added_atoms.size:
            plt.scatter(added_atoms[:, 0], added_atoms[:, 1], s=5, c="g", label="added atoms")

        plt.scatter(perfect_defects[:, 0], perfect_defects[:, 1], c="r", label="defects")
        # … rest of your quivers, adatoms, etc.
        plt.legend(loc="upper left", fontsize=8)
        plt.axis('off')
        plt.quiver(*center, b1[0], b1[1], color='g', angles='xy', scale_units='xy', scale=.1, label='Lattice Directions')
        plt.quiver(*center, b2[0], b2[1], color='g', angles='xy', scale_units='xy', scale=.1)
        plt.quiver(*center, b1_rot[0], b1_rot[1], color='orange', angles='xy', scale_units='xy', scale=.1, label='Same Layer Vectors')
        plt.quiver(*center, b2_rot[0], b2_rot[1], color='orange', angles='xy', scale_units='xy', scale=.1)
        plt.scatter(center[0], center[1], c="y", s=50)
        plt.legend(loc="upper left", fontsize=8)
        plt.axis('off')
    
    mask_dots = (perfect_dots[:, 0] >= 0) & (perfect_dots[:, 0] <= 512) & (perfect_dots[:, 1] >= 0) & (perfect_dots[:, 1] <= 512)
    mask_def = (perfect_defects[:, 0] >= 0) & (perfect_defects[:, 0] <= 512) & (perfect_defects[:, 1] >= 0) & (perfect_defects[:, 1] <= 512)
    length = np.linalg.norm(b1)

    return perfect_dots[mask_dots], perfect_defects[mask_def], [b1_rot, b2_rot], length





    
def snap_in_place(dots, defects, basis):
    invB   = np.linalg.inv(basis)
    pts    = np.vstack([dots, defects])
    labels = np.r_[np.zeros(len(dots), int), np.ones(len(defects), int)]

    frac   = pts @ invB
    idx    = np.round(frac).astype(int)

    idx, labels = _unique_and_fill(idx, frac, labels)  # keep labels aligned
    return idx @ basis, labels

def _unique_and_fill(idx, frac, labels):
    # idx : (N,2) int
    # labels : 0 - dot, 1 - defect
    uniq, cnt = np.unique(idx, axis=0, return_counts=True)
    keep  = np.ones(len(idx), bool)
    extra = []                # rows (dots only) that will be moved

    for u in uniq[cnt > 1]:
        rows       = np.where((idx == u).all(1))[0]
        def_rows   = rows[labels[rows] == 1]   # defects at this site
        dot_rows   = rows[labels[rows] == 0]   # dots   at this site

        # keep exactly one defect if present, otherwise one closest dot
        if def_rows.size:
            keep[def_rows] = False
            keep[def_rows[0]] = True          # leave first defect
            # dots at the same index become extras
            extra.extend(dot_rows)
        else:
            d = np.linalg.norm(frac[dot_rows] - u, axis=1)
            k = dot_rows[np.argmin(d)]        # keep closest dot
            keep[dot_rows] = False
            keep[k] = True
            extra.extend(r for r in dot_rows if r != k)

    # bounding box of all snapped points
    xmin, ymin = idx.min(0)
    xmax, ymax = idx.max(0)
    full = {(i, j) for i in range(xmin, xmax + 1)
                     for j in range(ymin, ymax + 1)}

    taken = {tuple(p) for p in idx[keep]}
    free  = np.array([p for p in full - taken], int)

    if free.size and extra:
        from scipy.spatial import cKDTree
        tree = cKDTree(free)
        for r in extra:
            _, pos = tree.query(frac[r])
            new    = free[pos]
            idx[r] = new
            # remove the chosen site
            free   = np.delete(free, pos, axis=0)
            if free.size:
                tree = cKDTree(free)

    return idx, labels


def refine_basis(points, cell,
                 dθ=0.05, spanθ=2.0,
                 ds=0.01, span_s=0.05):
    """Return the basis closest to hexagonal integers, penalizing interior missing sites."""
    a0, b0 = np.arctan2(cell[:,1], cell[:,0]) * 180/np.pi
    L0     = np.linalg.norm(cell[0])
    best_B, best_err = cell, np.inf
    w, h = 512, 512  # image size
    count = 0

    for da in np.arange(-spanθ, spanθ + 1e-9, dθ):
        for db in np.arange(-spanθ, spanθ + 1e-9, dθ):
            for s in np.arange(1 - span_s, 1 + span_s + 1e-9, ds):
                B = s * L0 * np.vstack([[np.cos(np.deg2rad(a0 + da)),
                                         np.sin(np.deg2rad(a0 + da))],
                                        [np.cos(np.deg2rad(b0 + db)),
                                         np.sin(np.deg2rad(b0 + db))]])
                invB = np.linalg.inv(B)
                coords = points @ invB
                recon = coords @ B

                if (recon[:,0].min() < -5 or recon[:,0].max() > w+5 or
                    recon[:,1].min() < -5 or recon[:,1].max() > h+5):
                    continue

                err = np.linalg.norm(coords - np.round(coords), axis=1).mean()
                count += 1
                #if count % 1000 == 0:
                    #print(f"Checked {count} combinations, current best err: {best_err:.4f}")

                if err < best_err:
                    best_err, best_B = err, B


    return best_B, best_err


def calculate_center(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    center_most_index = np.argmin(distances)
    center_point = points[center_most_index]
    return center_point


def grid_plot(b1_rot, b2_rot, image_arr, center, perfect_dots, med_len_x):
    b1_unit = b1_rot / np.linalg.norm(b1_rot)
    b2_unit = b2_rot / np.linalg.norm(b2_rot)

    dist = 2 * med_len_x * np.sin(np.pi/3)


    # Scale them to med_len_x spacing
    b1_step = b1_unit * dist
    b2_step = b2_unit * dist

    # Define how many lines to draw in each direction
    n_lines = 10  # adjust as needed

    # Generate line offsets centered around the center point
    offsets = np.arange(-n_lines//2, n_lines//2 + 1)

    # Get image bounds
    height, width = image_arr.shape
    img_diagonal = np.sqrt(width**2 + height**2)
    line_len = img_diagonal * .4 # ensure lines span the full image

    # Plot image
    plt.figure(figsize=(6,6))
    plt.imshow(image_arr, cmap="gray", origin="lower")

    # Plot lines along b1_rot direction
    for k in offsets:
        offset_point = center + k * b2_step  # move perpendicular to b1
        start = offset_point - b1_unit * line_len / 2
        end   = offset_point + b1_unit * line_len / 2
        plt.plot([start[0], end[0]], [start[1], end[1]], 'y-', linewidth=1)

    # Plot lines along b2_rot direction
    for k in offsets:
        offset_point = center + k * b1_step  # move perpendicular to b2
        start = offset_point - b2_unit * line_len / 2
        end   = offset_point + b2_unit * line_len / 2
        plt.plot([start[0], end[0]], [start[1], end[1]], 'y-', linewidth=1)

    # Optional: mark the center
    plt.scatter(perfect_dots[:,0], perfect_dots[:,1], s = 5, c = "red")
    plt.scatter(center[0], center[1], c = "y", s = 50)

    # Optional: show b1_rot and b2_rot
    plt.quiver(*center, b1_rot[0], b1_rot[1], color='b', angles='xy', scale_units='xy', scale=0.08)
    plt.quiver(*center, b2_rot[0], b2_rot[1], color='b', angles='xy', scale_units='xy', scale=0.08)

    plt.show()

def is_within_tolerance(point, other_points, tolerance=0.05):
    other_points = np.atleast_2d(other_points)
    if other_points.size == 0:
        return False
    distances = np.linalg.norm(other_points - point, axis=1)
    return np.any(distances <= tolerance)




def ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def remap_groups_by_overlap(prev_groups, curr_groups, tol=0.3, verbose = False):
    def to_array(group):
        arr = np.atleast_2d(np.array(group, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float64)
        return arr

    prev = [to_array(g) for g in prev_groups]
    curr = [to_array(g) for g in curr_groups]

    best_mapping = None

    for perm in permutations([0, 1, 2]):
        all_overlap = True
        for i in range(3):
            pg, cg = prev[i], curr[perm[i]]
            if len(pg) == 0 or len(cg) == 0:
                all_overlap = False
                break
            dists = np.linalg.norm(pg[:, None, :] - cg[None, :, :], axis=2)
            if not np.any(dists < tol):
                all_overlap = False
                break
        if all_overlap:
            best_mapping = perm
            break

    if best_mapping is None:
        if verbose:
            print("WARNING!!! No valid mapping found between groups.")
        return None
    return tuple(curr[best_mapping[i]] for i in range(3))

def get_points_in_block_3x3(center, points, block_boxes):
    for key, selector in block_boxes.items():
        if selector(np.array([center]))[0]:
            return selector(points)
    return np.zeros(len(points), dtype=bool)  # fallback mask

def get_block_bounding_boxes_3x3(points, margin_frac=0.05):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    x_edges = np.linspace(x_min, x_max, 4)
    y_edges = np.linspace(y_min, y_max, 4)
    x_margin = (x_max - x_min) * margin_frac
    y_margin = (y_max - y_min) * margin_frac

    blocks = {}
    for i in range(3):  # rows
        for j in range(3):  # cols
            key = f'B{i*3 + j + 1}'  # B1 through B9
            def box(p, xi=x_edges[j], xj=x_edges[j+1], yi=y_edges[i], yj=y_edges[i+1]):
                return (
                    (p[:, 0] >= xi - x_margin) & (p[:, 0] < xj + x_margin) &
                    (p[:, 1] >= yi - y_margin) & (p[:, 1] < yj + y_margin)
                )
            blocks[key] = box
    return blocks

def calculate_block_centers_3x3(points):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    x_edges = np.linspace(x_min, x_max, 4)
    y_edges = np.linspace(y_min, y_max, 4)

    centers = []
    for i in range(3):  # rows
        for j in range(3):  # columns
            mask = (
                (points[:, 0] >= x_edges[j]) & (points[:, 0] < x_edges[j+1]) &
                (points[:, 1] >= y_edges[i]) & (points[:, 1] < y_edges[i+1])
            )
            block_points = points[mask]
            if len(block_points) == 0:
                centers.append(None)
            else:
                centroid = np.mean(block_points, axis=0)
                idx = np.argmin(np.linalg.norm(block_points - centroid, axis=1))
                centers.append(block_points[idx])
    return np.array(centers, dtype=object)

def find_layer_new(perfect_dots, defects, rot_angles, med_len_x, *,
                   plot=True, verbose=False,
                   phase_correct=True, edge_margin_k = .8):
    labels     = ['atom'] * len(perfect_dots) + ['defect'] * len(defects)
    all_points = np.vstack([perfect_dots, defects]).astype(np.float64)

    df = pd.DataFrame(all_points, columns=['x', 'y'])
    df['label'] = labels
    df = df.reset_index(drop=True)

    b1_rot, b2_rot = rot_angles
    b1_unit = (b1_rot / np.linalg.norm(b1_rot)).astype(np.float64)
    b2_unit = (b2_rot / np.linalg.norm(b2_rot)).astype(np.float64)
    dist    = 2 * med_len_x * np.sin(np.pi / 3)
    B_scaled = np.column_stack((b1_unit * dist, b2_unit * dist)).astype(np.float64)
    T_scaled_to_std = np.linalg.inv(B_scaled).astype(np.float64)

    min_x, min_y = df[['x','y']].min().astype(np.float64)
    max_x, max_y = df[['x','y']].max().astype(np.float64)

    # bias rounding toward the origin in scaled space
    def biased_round(arr, eps=1e-6):
        return np.floor(arr + 0.5 - eps*np.sign(arr)).astype(np.int64)

    def closest_per_cell(idxs, origin):
        if len(idxs) == 0:
            return np.array([], dtype=int)

        sub    = df.loc[idxs]
        pts    = sub[['x','y']].to_numpy(dtype=np.float64, copy=False)
        origin = np.asarray(origin, dtype=np.float64).reshape(1, 2)
        rel    = pts - origin

        scaled0 = (T_scaled_to_std @ rel.T).T.astype(np.float64)

        if phase_correct:
            frac = scaled0 - np.rint(scaled0)
            phase = frac.mean(axis=0)
        else:
            phase = np.zeros(2, dtype=np.float64)

        scaled  = scaled0 - phase
        rounded = biased_round(scaled)
        std     = (B_scaled @ rounded.T).T + origin
        d2      = ((pts - std)**2).sum(axis=1)

        best = {}
        best_d2 = {}
        for i_local, key in enumerate(map(tuple, rounded)):
            gid = sub.index[i_local]
            if (key not in best) or (d2[i_local] < best_d2[key]):
                best[key]   = gid
                best_d2[key]= d2[i_local]

        if not best:
            return np.array([], dtype=int)
        return np.fromiter(best.values(), dtype=int)

    mean_ctr = df[['x','y']].to_numpy(dtype=np.float64).mean(axis=0)
    ctr_idx  = int(np.argmin(((df[['x','y']].to_numpy(dtype=np.float64) - mean_ctr)**2).sum(axis=1)))
    ctr      = df.loc[ctr_idx, ['x','y']].to_numpy(dtype=np.float64)

    all_idx = np.arange(len(df), dtype=int)
    out = df.copy()
    out['group'] = 0

    group_color = {1: 'fuchsia', 2: 'dodgerblue', 3: 'midnightblue', 0: '0.8'}
    label_size  = {'atom': 30, 'defect': 90}

    if plot:
        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(2, 2, height_ratios=[1,1], width_ratios=[1,1])
        ax0 = fig.add_subplot(gs[0,0]); ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[1,0]); ax3 = fig.add_subplot(gs[1,1])
        for ax in (ax0, ax1, ax2, ax3):
            ax.set_aspect('equal'); ax.axis('off')

        for lab in out['label'].unique():
            m = out['label'] == lab
            ax0.scatter(out.loc[m,'x'], out.loc[m,'y'], s=label_size[lab], c='0.35', edgecolors='none', label=lab)
        ax0.scatter([ctr[0]], [ctr[1]], s=140, marker='*', edgecolors='k', facecolors='gold')
        ax0.set_title('Raw points (+ mean-center star)')
        ax0.legend(frameon=False, loc='upper right')

    g1_idx = closest_per_cell(all_idx, ctr)
    out.loc[g1_idx, 'group'] = 1
    unassigned = np.setdiff1d(all_idx, g1_idx, assume_unique=False)

    if plot:
        ax1.set_title('Step 1: assign Group 1')
        for lab in out['label'].unique():
            m1 = (out.index.isin(g1_idx)) & (out['label']==lab)
            m0 = (~out.index.isin(g1_idx)) & (out['label']==lab)
            if m0.any(): ax1.scatter(out.loc[m0,'x'], out.loc[m0,'y'], s=label_size[lab], c='0.85', edgecolors='none')
            if m1.any(): ax1.scatter(out.loc[m1,'x'], out.loc[m1,'y'], s=label_size[lab], c=group_color[1], edgecolors='none')
        ax1.scatter([ctr[0]], [ctr[1]], s=120, marker='+', c='k')

    if unassigned.size:
        d2 = ((df.loc[unassigned, ['x','y']].to_numpy(dtype=np.float64) - ctr)**2).sum(axis=1)
        center2 = df.loc[unassigned[int(np.argmin(d2))], ['x','y']].to_numpy(dtype=np.float64)
        g2_idx = closest_per_cell(unassigned, center2)
        out.loc[g2_idx, 'group'] = 2
        unassigned = np.setdiff1d(unassigned, g2_idx, assume_unique=False)
    else:
        g2_idx = np.array([], dtype=int)

    if plot:
        ax2.set_title('Step 2: assign Group 2')
        for lab in out['label'].unique():
            m2 = (out.index.isin(g2_idx)) & (out['label']==lab)
            m1 = (out.index.isin(g1_idx)) & (out['label']==lab)
            m0 = (~out.index.isin(np.r_[g1_idx, g2_idx])) & (out['label']==lab)
            if m0.any(): ax2.scatter(out.loc[m0,'x'], out.loc[m0,'y'], s=label_size[lab], c='0.85', edgecolors='none')
            if m1.any(): ax2.scatter(out.loc[m1,'x'], out.loc[m1,'y'], s=label_size[lab], c=group_color[1], edgecolors='none')
            if m2.any(): ax2.scatter(out.loc[m2,'x'], out.loc[m2,'y'], s=label_size[lab], c=group_color[2], edgecolors='none')
        ax2.scatter([center2[0]], [center2[1]], s=120, marker='+', c='k')

    g3_idx = unassigned
    out.loc[g3_idx, 'group'] = 3
    # --- Reassign edge coords to nearest group via re-snap per-group origin ---
    edge_margin = float(edge_margin_k) * float(med_len_x)
    edge_mask = (
        (out['x'] - min_x < edge_margin) |
        (max_x - out['x'] < edge_margin) |
        (out['y'] - min_y < edge_margin) |
        (max_y - out['y'] < edge_margin)
    ).to_numpy()

    # pick a middle coord (closest to median) per group
    group_origins = {}
    phase_by_group = {}
    for g in (1, 2, 3):
        gi = out.index[out['group'] == g].to_numpy()
        if gi.size == 0:
            continue
        ptsg = out.loc[gi, ['x','y']].to_numpy(dtype=np.float64)
        med  = np.median(ptsg, axis=0)
        idx_med = gi[np.argmin(((ptsg - med)**2).sum(axis=1))]
        origin_g = out.loc[idx_med, ['x','y']].to_numpy(dtype=np.float64)
        group_origins[g] = origin_g

        if phase_correct and ptsg.shape[0] > 0:
            sc0 = (T_scaled_to_std @ (ptsg - origin_g).T).T
            frac = sc0 - np.rint(sc0)
            phase_by_group[g] = frac.mean(axis=0)
        else:
            phase_by_group[g] = np.zeros(2, dtype=np.float64)

    edge_idxs = out.index[edge_mask].to_numpy()
    if edge_idxs.size:
        P = out.loc[edge_idxs, ['x','y']].to_numpy(dtype=np.float64)
        best_g  = np.zeros(edge_idxs.size, dtype=int)
        best_d2 = np.full(edge_idxs.size, np.inf)

        for g, origin_g in group_origins.items():
            sc  = (T_scaled_to_std @ (P - origin_g).T).T - phase_by_group[g]
            rnd = biased_round(sc)
            std = (B_scaled @ rnd.T).T + origin_g
            d2  = ((P - std)**2).sum(axis=1)

            take = d2 < best_d2
            best_d2[take] = d2[take]
            best_g[take]  = g

        out.loc[edge_idxs, 'group'] = best_g



    defect_counts = out[out['label']=='defect'].groupby('group').size().reindex([1,2,3], fill_value=0)
    sorted_groups = defect_counts.sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(sorted_groups, start=1)}
    out['group'] = out['group'].map(remap)

    if plot:
        ax3.set_title('Final (groups remapped by defect count)')
        for g in [1,2,3]:
            for lab in out['label'].unique():
                m = (out['group']==g) & (out['label']==lab)
                if m.any():
                    ax3.scatter(out.loc[m,'x'], out.loc[m,'y'], s=label_size[lab], c=group_color[g], edgecolors='none', label=f'G{g}-{lab}')
        handles, labels_ = ax3.get_legend_handles_labels()
        by = {l:h for h,l in zip(handles, labels_)}
        ax3.legend(by.values(), by.keys(), frameon=False, loc='upper right')
        plt.tight_layout()
        plt.show()

        # Rounding diagnostics in scaled coordinates for Group 1 origin
        sub  = df.loc[all_idx]
        pts  = sub[['x','y']].to_numpy(dtype=np.float64, copy=False)
        rel  = pts - ctr
        sc0  = (T_scaled_to_std @ rel.T).T.astype(np.float64)
        if phase_correct:
            phase = (sc0 - np.rint(sc0)).mean(axis=0)
        else:
            phase = np.zeros(2, dtype=np.float64)
        sc   = sc0 - phase
        rnd  = biased_round(sc)
        figR = plt.figure(figsize=(6,6))
        axR  = figR.add_subplot(111)
        axR.set_aspect('equal')
        axR.set_title('Rounding in scaled lattice coords (arrows: scaled → rounded)')
        axR.scatter(sc[:,0], sc[:,1], s=8, alpha=0.6)
        axR.scatter(rnd[:,0], rnd[:,1], s=16, marker='s', alpha=0.7)
        for i in range(len(sc)):
            axR.plot([sc[i,0], rnd[i,0]], [sc[i,1], rnd[i,1]], linewidth=0.5, alpha=0.4)
        axR.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout(); plt.show()

    if verbose:
        pass  # no edge rejections anymore

    return out




def find_layer_updated(perfect_dots, defects, rot_angles, med_len_x, tol=0.25, plot=True, verbose=False):

    centers = calculate_block_centers_3x3(perfect_dots)
    block_boxes = get_block_bounding_boxes_3x3(perfect_dots)

    labels = ['atom'] * len(perfect_dots) + ['defect'] * len(defects)
    all_points = np.vstack([perfect_dots, defects]).astype(np.float64)

    df = pd.DataFrame(all_points, columns=['x', 'y'])
    df['label'] = labels

    
    def transform_and_filter(df_block, origin):
        rel     = df_block[['x', 'y']].values - origin
        scaled  = (T_scaled_to_std @ rel.T).T.astype(np.float64)
        rounded = np.round(scaled).astype(int)
        std     = (B_scaled @ rounded.T).T + origin

        d = ((df_block[['x','y']].values - std)**2).sum(axis=1)

        best = {}
        for i, k in enumerate(map(tuple, rounded)):
            if k not in best or d[i] < d[best[k]]:
                best[k] = i

        keep_idx = list(best.values())
        return df_block.iloc[keep_idx].reset_index(drop=True)





    b1_rot, b2_rot = rot_angles
    b1_unit = b1_rot / np.linalg.norm(b1_rot)
    b2_unit = b2_rot / np.linalg.norm(b2_rot)
    dist = 2 * med_len_x * np.sin(np.pi / 3)
    b1_step = b1_unit * dist
    b2_step = b2_unit * dist
    B_scaled = np.column_stack((b1_step, b2_step))
    T_scaled_to_std = np.linalg.inv(B_scaled)

    if plot and verbose:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        at = df['label'] == 'atom'
        de = df['label'] == 'defect'
        ax.scatter(df.loc[at, 'x'], df.loc[at, 'y'], s=5, c='0.75', alpha=0.7, edgecolors='none')
        ax.scatter(df.loc[de, 'x'], df.loc[de, 'y'], s=30, c='cyan', alpha=0.9, edgecolors='none')
        cxy = np.array([c for c in centers if c is not None])
        if len(cxy):
            ax.scatter(cxy[:, 0], cxy[:, 1], s=60, c='k', marker='+', linewidths=1.5)
        ax.set_aspect('equal')
        ax.set_title('Raw points with block centers')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    block_results = []

    for center in centers:
        if center is None:
            block_results.append(pd.DataFrame(columns=['x', 'y', 'label', 'group']))
            continue

        in_block = get_points_in_block_3x3(center, df[['x', 'y']].values, block_boxes)
        df_block = df[in_block].copy()

        g1_df = transform_and_filter(df_block, center)
        g1_df['group'] = 1

        dist_to_center = np.linalg.norm(df[['x', 'y']].values.astype(np.float64) - np.array(center, dtype=np.float64), axis=1)
        sorted_indices = np.argsort(dist_to_center)

        g1_coords = g1_df[['x', 'y']].values.astype(np.float64)

        new_center = None
        for idx in sorted_indices[1:]:
            candidate = df[['x', 'y']].values[idx].astype(np.float64)
            if not np.any(np.all(np.isclose(candidate, g1_coords, atol=1e-3), axis=1)):
                new_center = candidate
                break

        if new_center is not None:
            in_block2 = get_points_in_block_3x3(new_center, df[['x', 'y']].values, block_boxes)
            df_block2 = df[in_block2].copy()
            g2_df = transform_and_filter(df_block2, new_center)
            g2_df['group'] = 2
        else:
            g2_df = pd.DataFrame(columns=['x', 'y', 'label', 'group'])

        g12_coords = pd.concat([g1_df, g2_df])[['x', 'y']].values if not g2_df.empty else g1_df[['x', 'y']].values
        g3_df = df_block[~df_block.index.isin(pd.concat([g1_df, g2_df]).index)].copy()
        g3_df['group'] = 3


        if plot and verbose:
            group_color = {1: 'fuchsia', 2: 'dodgerblue', 3: 'midnightblue'}
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

            at_blk = df_block['label'] == 'atom'
            de_blk = df_block['label'] == 'defect'
            axL.scatter(df_block.loc[at_blk, 'x'], df_block.loc[at_blk, 'y'], s=5, c='0.8', edgecolors='none')
            axL.scatter(df_block.loc[de_blk, 'x'], df_block.loc[de_blk, 'y'], s=40, c='cyan', edgecolors='none')
            axL.scatter(center[0], center[1], s=70, marker='x', c='k', linewidths=1.5)
            if new_center is not None:
                axL.scatter(new_center[0], new_center[1], s=70, marker='x', c='orange', linewidths=1.5)
            axL.set_title('Block input')
            axL.set_aspect('equal'); axL.axis('off')

            if not g1_df.empty:
                axR.scatter(g1_df['x'], g1_df['y'], s=12, c=group_color[1], edgecolors='none', label='g1')
            if not g2_df.empty:
                axR.scatter(g2_df['x'], g2_df['y'], s=12, c=group_color[2], edgecolors='none', label='g2')
            if not g3_df.empty:
                axR.scatter(g3_df['x'], g3_df['y'], s=12, c=group_color[3], edgecolors='none', label='g3')
            axR.set_title('Snapped groups')
            axR.set_aspect('equal'); axR.axis('off')
            plt.show()

        block_df = pd.concat([g1_df, g2_df, g3_df], ignore_index=True)
        block_results.append(block_df)

    all_blocks = pd.concat(block_results, ignore_index=True)

    ref_block = block_results[4]
    ref_groups = [ref_block[ref_block['group'] == i][['x', 'y']].values for i in [1, 2, 3]]

    group_remaps = []
    for block_df in block_results:
        if block_df.empty:
            group_remaps.append(block_df)
            continue
        curr_groups = [block_df[block_df['group'] == i][['x', 'y']].values for i in [1, 2, 3]]
        res = remap_groups_by_overlap(ref_groups, curr_groups, tol=tol, verbose=verbose)
        if res is None:
            group_remaps.append(block_df)
            continue
        r1, r2, r3 = res
        mapping = {}
        for new_g, r in zip([1, 2, 3], [r1, r2, r3]):
            for pt in map(tuple, r):
                mapping[pt] = new_g
        coords = block_df[['x', 'y']].apply(tuple, axis=1)
        block_df['group'] = coords.map(mapping).fillna(block_df['group'])
        group_remaps.append(block_df)

    all_blocks = pd.concat(group_remaps, ignore_index=True)
    coords = all_blocks[['x', 'y']].to_numpy()
    db = DBSCAN(eps=0.1, min_samples=1).fit(coords)
    all_blocks = all_blocks.groupby(db.labels_).first().reset_index(drop=True)

    defect_counts = (all_blocks[all_blocks['label'] == 'defect']['group']
                     .value_counts().reindex([1, 2, 3], fill_value=0))
    sorted_groups = defect_counts.sort_values().index.tolist()
    group_remap = {old: new for new, old in enumerate(sorted_groups, start=1)}
    all_blocks['group'] = all_blocks['group'].map(group_remap).fillna(all_blocks['group'])

    if plot:
        group_color = {1: 'fuchsia', 2: 'dodgerblue', 3: 'midnightblue'}
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 0])
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        for group in [1, 2, 3]:
            for label in all_blocks['label'].unique():
                mask = (all_blocks['group'] == group) & (all_blocks['label'] == label)
                size_array = 100 if label == 'defect' else 25
                ax1.scatter(all_blocks.loc[mask, 'x'], all_blocks.loc[mask, 'y'],
                            c=group_color[group], edgecolors='none', s=size_array, alpha=1)
        ax1.set_aspect('equal')
        ax1.axis('off')

        group_counts = (all_blocks[all_blocks['label'] == 'defect']['group']
                        .value_counts().reindex([1, 2, 3], fill_value=0))
        ax2.bar(['Group 1', 'Group 2', 'Group 3'],
                group_counts.values,
                color=['fuchsia', 'dodgerblue', 'midnightblue'])
        ax2.set_ylabel('Number of Defects')
        ax2.set_title('Defects by Group')

        plt.tight_layout()
        plt.show()

    return all_blocks





def find_layer2(perfect_dots, defects, rot_angles, med_len_x, adatom_points=None, adatoms=None, tol=0.25, plot=True, contour_mask=None, adatom_hist=True, verbose=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from skimage import measure
    from matplotlib.patches import Polygon

    centers = calculate_block_centers_3x3(perfect_dots)
    block_boxes = get_block_bounding_boxes_3x3(perfect_dots)

    labels = ['atom'] * len(perfect_dots) + ['defect'] * len(defects)
    all_points = np.vstack([perfect_dots, defects]).astype(np.float64)
    if adatoms is not None:
        labels += ['adatom'] * len(adatoms)
        all_points = np.vstack([all_points, adatoms]).astype(np.float64)

    df = pd.DataFrame(all_points, columns=['x', 'y'])
    df['label'] = labels

    def transform_and_filter(df_block, origin):
        rel = df_block[['x', 'y']].values - origin
        scaled = (T_scaled_to_std @ rel.T).T.astype(np.float64)
        rounded = np.round(scaled)
        std = (B_scaled @ rounded.T).T + origin
        df_snapped = df_block.copy()
        df_snapped[['x', 'y']] = std
        return df_snapped


    b1_rot, b2_rot = rot_angles
    b1_unit = b1_rot / np.linalg.norm(b1_rot)
    b2_unit = b2_rot / np.linalg.norm(b2_rot)
    dist = 2 * med_len_x * np.sin(np.pi / 3)
    b1_step = b1_unit * dist
    b2_step = b2_unit * dist
    B_scaled = np.column_stack((b1_step, b2_step))
    T_scaled_to_std = np.linalg.inv(B_scaled)


    block_results = []

    for center in centers:
        if center is None:
            block_results.append(pd.DataFrame(columns=['x', 'y', 'label', 'group']))
            continue

        in_block = get_points_in_block_3x3(center, df[['x', 'y']].values, block_boxes)
        df_block = df[in_block].copy()
        g1_df = transform_and_filter(df_block, center)
        g1_df['group'] = 1

        dist_to_center = np.linalg.norm(df[['x', 'y']].values.astype(np.float64) - np.array(center, dtype=np.float64), axis=1)
        sorted_indices = np.argsort(dist_to_center)

        g1_coords = g1_df[['x', 'y']].values.astype(np.float64)
        
        for idx in sorted_indices[1:]:
            candidate = df[['x', 'y']].values[idx].astype(np.float64)
            if not np.any(np.all(np.isclose(candidate, g1_coords, atol=1e-3), axis=1)):
                new_center = candidate
                break
        else:
            new_center = None

        in_block = get_points_in_block_3x3(new_center, df[['x', 'y']].values, block_boxes)
        df_block = df[in_block].copy()
        g2_df = transform_and_filter(df_block, new_center)
        g2_df['group'] = 2

        g12_coords = pd.concat([g1_df, g2_df])[['x', 'y']].values
        mask_g3 = ~np.array([is_within_tolerance(pt, g12_coords, tol) for pt in df[['x', 'y']].values])
        g3_df = df[mask_g3].copy()
        in_block = get_points_in_block_3x3(center, g3_df[['x', 'y']].values, block_boxes)
        g3_df = g3_df[in_block].copy()
        g3_df['group'] = 3

        block_df = pd.concat([g1_df, g2_df, g3_df], ignore_index=True)
        block_results.append(block_df)

    all_blocks = pd.concat(block_results, ignore_index=True)

    ref_block = block_results[4]  # center
    ref_groups = [ref_block[ref_block['group'] == i][['x', 'y']].values for i in [1, 2, 3]]

    group_remaps = []
    for block_df in block_results:
        if block_df.empty:
            group_remaps.append(block_df)
            continue
        curr_groups = [block_df[block_df['group'] == i][['x', 'y']].values for i in [1, 2, 3]]
        res = remap_groups_by_overlap(ref_groups, curr_groups, tol=tol,
                                      verbose=verbose)
        if res is None:
            group_remaps.append(block_df)
            continue
        
        r1, r2, r3 = res               
        mapping = {}
        for new_g, r in zip([1, 2, 3], [r1, r2, r3]):
            for pt in map(tuple, r):
                mapping[pt] = new_g
        coords = block_df[['x', 'y']].apply(tuple, axis=1)
        block_df['group'] = coords.map(mapping).fillna(block_df['group'])
        group_remaps.append(block_df)

    all_blocks = pd.concat(group_remaps, ignore_index=True)
    coords = all_blocks[['x', 'y']].to_numpy()
    db = DBSCAN(eps=0.1, min_samples=1).fit(coords)
    all_blocks = all_blocks.groupby(db.labels_).first().reset_index(drop=True)

    if adatom_points is not None:
        adatom_points = np.array(adatom_points)
        coords = all_blocks[['x', 'y']].values
        coords = np.asarray(coords, dtype=np.float64)
        adatom_points = np.asarray(adatom_points, dtype=np.float64)
        diff = coords[:, None, :] - adatom_points[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))

        near_adatom_mask = (dists < 3).any(axis=1)
        all_blocks['near_adatom'] = near_adatom_mask
    else:
        all_blocks['near_adatom'] = False

    defect_counts = (all_blocks[all_blocks['label'] == 'defect']['group'].value_counts().reindex([1, 2, 3], fill_value=0) )
    sorted_groups = defect_counts.sort_values().index.tolist()
    group_remap = {old: new for new, old in enumerate(sorted_groups, start=1)}
    all_blocks['group'] = all_blocks['group'].map(group_remap).fillna(all_blocks['group'])

    if plot:
        color_map = {'atom': 'k', 'defect': 'cyan', 'adatom': 'pink'}
        group_color = {1: 'fuchsia', 2: 'dodgerblue', 3: 'midnightblue'}
        fig = plt.figure(figsize=(14, 10))
        if adatom_hist:
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
        else:
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 0])
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])

        all_blocks.loc[all_blocks['near_adatom'], 'label'] = 'adatom'


        if contour_mask is not None:
            mask = contour_mask
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

            for color in unique_colors:
                if np.all(color == 0):
                    continue
                region = np.all(mask == color, axis=-1)

                # 🔺 Dilate the region to make it blob-like
                region_dilated = binary_dilation(region, iterations=5)

                contours = measure.find_contours(region_dilated, 0.5)
                for contour in contours:
                    poly = Polygon(
                        contour[:, [1, 0]],
                        closed=True,
                        facecolor="black",
                        linewidth=1.5,
                        alpha=0.6,
                        zorder=1
                    )
                    ax1.add_patch(poly)

        for group in [1, 2, 3]:
            for label in all_blocks['label'].unique():
                mask = (all_blocks['group'] == group) & (all_blocks['label'] == label)
                is_near = all_blocks.loc[mask, 'near_adatom']
                size_array = np.where(is_near, 200, (100 if label == 'defect' else 25))
                marker = 'x' if label == 'adatom' else 'o'
                if label == 'adatom':
                    ax1.scatter(all_blocks.loc[mask, 'x'], all_blocks.loc[mask, 'y'], label=f'{label} g{group}', color=group_color[group], s=size_array, alpha=1, marker=marker)
                else:
                    ax1.scatter(all_blocks.loc[mask, 'x'], all_blocks.loc[mask, 'y'], label=f'{label} g{group}', c=group_color[group], edgecolors='none', s=size_array, alpha=1, marker=marker)

        ax1.set_aspect('equal')
        #ax1.legend()
        ax1.axis('off')

        group_counts = all_blocks[all_blocks['label'] == 'defect']['group'].value_counts().reindex([1, 2, 3], fill_value=0)
        ax2.bar(['Group 1', 'Group 2', 'Group 3'], group_counts.values, color=['fuchsia', 'dodgerblue', 'midnightblue'])
        ax2.set_ylabel('Number of Defects')
        ax2.set_title('Defects by Group')

        if adatom_hist:
            near_adatom_counts = all_blocks[all_blocks['near_adatom']]['group'].value_counts().reindex([1, 2, 3], fill_value=0)
            ax3.bar(['Group 1', 'Group 2', 'Group 3'], near_adatom_counts.values, color=['fuchsia', 'dodgerblue', 'midnightblue'])
            ax3.set_ylabel('Number of Adatoms')
            ax3.set_title('Adatoms by Group')

        plt.tight_layout()
        plt.show()

    return all_blocks


def distance_defect(image, x, y , resolution, pixel_thresh = None, min_area = None, graph_colors = "yes", colormap_name = "Set1") :
    
    # initialize a mesh grid
    x_max, y_max = image.shape
    xx, yy = np.meshgrid(
    np.arange(0, x_max, resolution), 
    np.arange(0, y_max, resolution))
    neighborhood_radius = 10
    points = np.vstack((x, y)).T

    #  create a KD tree and find avg distance between 6 nearest dots (7 to exclude dot itself)
    tree = cKDTree(points)
    dists, indices = tree.query(points,  k=7)
    dists = dists[:,1:]
    indices = indices[:,1:]
    avg_dist = mode(mode(dists, axis=1)[0])[0]  # take mode because there are outliers - points near vacancies
    
    if pixel_thresh is None:
        pixel_thresh = .8*avg_dist

    if min_area is None:
        #min_area = 1
        min_area = .2*pixel_thresh**2
    
    distances = np.full(xx.shape, np.inf)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            nearby_indices = tree.query_ball_point([xx[i, j], yy[i, j]], neighborhood_radius)
            
            if nearby_indices:  
                nearby_dists = np.sqrt((xx[i, j] - x[nearby_indices]) ** 2 + (yy[i, j] - y[nearby_indices]) ** 2)
                distances[i, j] = np.min(nearby_dists)  

    pixel_threshold = pixel_thresh
    edge_exclusion = 20

    point_mask = distances > pixel_threshold
    edge_mask = (
        (xx > 0 + edge_exclusion) & (xx < x_max - edge_exclusion) &
        (yy > 0 + edge_exclusion) & (yy < y_max - edge_exclusion)
    )
    final_mask = point_mask & edge_mask
    

    def contour_area(contour):
        x = contour[:, 1] * resolution  # Convert from grid to plot coordinates
        y = contour[:, 0] * resolution
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    contours = measure.find_contours(final_mask)
    filtered_contours = []
    areas = []  
    for i, contour in enumerate(contours):
        area = contour_area(contour)
        if area > min_area:
            filtered_contours.append((contour, area))
            areas.append(area)


    for contour, area in filtered_contours:
        plt.plot(contour[:, 1] * resolution, contour[:, 0] * resolution, color='red', linewidth=2)
    plt.imshow(image, cmap='gray', origin="lower")
    plt.scatter(x, y, color='yellow', s=0.5, alpha = .3)    
    plt.show()
    plt.close()

    if graph_colors:
        data = graph_color_area(filtered_contours, areas, x, y, image, resolution, colormap_name, avg_dist)
        return data



def graph_color_area(filtered_contours, areas, x, y, image, resolution, colormap_name, avg_dist ):

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2, height_ratios=[1, 2, 1], width_ratios=[1, 2], figure=fig)

    single_vacancy = 3/10*3.14* avg_dist**2
    divacancy = 6/10*3.14 * avg_dist**2
    bin_edges = sorted([0, single_vacancy, divacancy, max(areas) + 1])
    colormap = plt.get_cmap(colormap_name)
    num_bins = 5 
    hist, bins = np.histogram(areas, bins=bin_edges)

    ax1 = fig.add_subplot(gs[1, 0])
    for i in range(len(bins) - 1):
        bin_color = colormap(i % colormap.N) 
        width = bins[i+1]-bins[i]
        ax1.bar(bins[i], hist[i], width = width, align = 'edge', color=bin_color, alpha=0.9, edgecolor='white')
    ax1.set_title("Area histogram")

    data = {
    'Bin Start': bins[:-1],  
    'Bin End': bins[1:],     
    'Count': hist            
    }

    contour_mask = np.zeros((image.shape[0], image.shape[1], 3))



    ax2 = fig.add_subplot(gs[:, 1])
    for contour, area in filtered_contours:
        bin_index = np.digitize(area, bins) - 1
        bin_index = min(bin_index, len(bins) - 2) 
        contour_color = colormap(bin_index % colormap.N)[:3]   
        rr, cc = contour[:, 0].astype(int) * resolution, contour[:, 1].astype(int) * resolution
        rr, cc = draw.polygon(rr, cc, contour_mask.shape[:2])
        contour_mask[rr, cc] = contour_color
        ax2.fill(contour[:, 1] * resolution, contour[:, 0] * resolution, color=contour_color, alpha=0.6)

    ax2.imshow(image, cmap='gray', origin="lower")
    ax2.scatter(x, y, color='yellow', s=0.5, alpha = .5)
    ax2.set_title(f"defects, resolution= {resolution}")
    plt.show()
    plt.close()

    return data, contour_mask
