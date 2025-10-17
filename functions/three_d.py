
import pickle
import pandas as pd
import numpy as np
import ast
from scipy.spatial import Delaunay
import pickle, math, pathlib
import pandas as pd
import networkx as nx
import atomman as am
import os
from . import layers as layers  
from . import finding_defects as df  
import matplotlib.pyplot as plt

def flatten_cat(tags):
    # if tags is a string like "['same_layer_3','multilayer']", parse it
    if isinstance(tags, str):
        try:
            tags = ast.literal_eval(tags)
        except Exception:
            tags = [tags]

    # now tags is guaranteed to be a list
    # 1) nanopore or isolated stand alone
    if 'nanopore' in tags:
        return ['nanopore']
    if 'isolated vacancy' in tags:
        return ['isolated vacancy']

    # 2) keep any non-same_layer tags (e.g. inter_layer, multilayer)
    new_tags = [t for t in tags if not t.startswith('same_layer')]

    # 3) collapse any same_layer_* into one within_layer
    if any(t.startswith('same_layer') for t in tags):
        new_tags.append('within_layer')

    # 4) if nothing left (e.g. only same_layer), default to within_layer
    if not new_tags:
        new_tags = ['within_layer']

    return new_tags



def classify_vacancies(all_blocks, img=None):
    # ── split atoms / vacancies ──────────────────────────────────────────
    atom_pts = all_blocks.query("label=='atom'")[['x', 'y']].to_numpy()
    def_df   = all_blocks.query("label=='defect'").copy()
    vac_pts  = def_df[['x', 'y']].to_numpy()
    grp_arr  = def_df['group'].to_numpy()

    # defaults
    all_blocks['category']   = 'atom'
    all_blocks['cluster_id'] = -1

    # ── vacancy sizes (your helper) ──────────────────────────────────────
    glob_sz, _, _ = df.cluster_by_size(atom_pts, vac_pts, img, plot=False)

    # ── global triangulation (cross-layer links) ────────────────────────
    coords = np.vstack([atom_pts, vac_pts])
    labels = np.hstack([np.zeros(len(atom_pts)), np.ones(len(vac_pts))])
    tri    = Delaunay(coords)
    edges  = {tuple(sorted((i, j)))
              for s in tri.simplices
              for i, j in zip(s, s[[1, 2, 0]])}

    g_def   = nx.Graph()
    for i_all, j_all in edges:
        if labels[i_all] == labels[j_all] == 1:       # both vacancies
            i = i_all - len(atom_pts)
            j = j_all - len(atom_pts)
            g_def.add_edge(i, j)

    # map each vacancy → its component
    def_cluster_map = {}
    for comp in nx.connected_components(g_def):
        for idx in comp:
            def_cluster_map[idx] = comp

    # ── initial categories (nanopore / complex) ─────────────────────────
    cat = [[] for _ in range(len(def_df))]
    for i, sz in enumerate(glob_sz):
        comp   = def_cluster_map.get(i, {i})
        layers = {grp_arr[j] for j in comp}
        if sz >= 3 and layers == {1, 2, 3}:
            cat[i].append('nanopore')
        elif sz >= 2 and len(layers) > 1:
            cat[i].append('complex')

    # ── build master graph g_total (cross + same-layer) ─────────────────
    g_total = nx.Graph()
    g_total.add_nodes_from(range(len(def_df)))
    g_total.add_edges_from(g_def.edges())

    loc_sz_map = {}
    pending = [i for i, c in enumerate(cat) if 'nanopore' not in c]

    for g in np.unique(grp_arr[pending]):
        pos = [i for i in pending if grp_arr[i] == g]
        if len(pos) < 2:
            for i in pos:
                loc_sz_map[i] = 1
            continue

        atom_layer = all_blocks.query("label=='atom' and group==@g")[['x', 'y']].to_numpy()

        coords_l = np.vstack([atom_layer, vac_pts[pos]])
        lbls_l   = np.hstack([np.zeros(len(atom_layer)), np.ones(len(pos))])
        tri_l    = Delaunay(coords_l)
        edges_l  = {tuple(sorted((i, j)))
                    for s in tri_l.simplices
                    for i, j in zip(s, s[[1, 2, 0]])}

        shift = len(atom_layer)
        layer_edges = [(pos[i - shift], pos[j - shift])
                       for i, j in edges_l if lbls_l[i] == lbls_l[j] == 1]

        g_total.add_edges_from(layer_edges)

        g_layer = nx.Graph(layer_edges)
        for comp in nx.connected_components(g_layer):
            if any('complex' in cat[k] for k in comp):
                for k in comp:
                    if 'complex' not in cat[k]:
                        cat[k].append('complex')
            for k in comp:
                loc_sz_map[k] = len(comp)
        for k in pos:
            loc_sz_map.setdefault(k, 1)

    # ── convert complex → pit / sandwich where appropriate ──────────────
    for comp in nx.connected_components(g_def):
        if len(comp) == 2 and all('complex' in cat[k] for k in comp):
            if all(loc_sz_map.get(k, 1) == 1 for k in comp):
                layers = {grp_arr[k] for k in comp}
                tag = 'pit' if 1 in layers else 'sandwich'
                for k in comp:
                    cat[k] = [tag]

    # ── surface / middle clusters, isolated vacancies ───────────────────
    for i, lsz in loc_sz_map.items():
        if lsz > 1 and not cat[i]:
            cat[i].append('middle_cluster' if grp_arr[i] == 1 else 'surface_cluster')

    for i, c in enumerate(cat):
        if not c:
            cat[i].append('isolated vacancy')

    # ── write back categories ───────────────────────────────────────────
    all_blocks.loc[def_df.index, 'category'] = [c[0] for c in cat]

    # ── connectivity IDs ────────────────────────────────────────────────
    cluster_id = -np.ones(len(def_df), dtype=int)
    for cid, comp in enumerate(nx.connected_components(g_total)):
        for idx in comp:
            cluster_id[idx] = cid
    all_blocks.loc[def_df.index, 'cluster_id'] = cluster_id

    return all_blocks



def make_df(
    PERCENTS        = ["5", "9_1", "12_5"],
    DATA_DIR        = pathlib.Path("/Users/gguinan/Library/CloudStorage/OneDrive-NREL/",
                                "spring2025code/for_github2/data"),
    CLUSTER_COLORS  = ["orange", "#4daf4a", "#f781bf", "#e41a1c"]
):


    COLOR2SIZE      = {c: i + 1 for i, c in enumerate(CLUSTER_COLORS)}
    # ── MAIN LOOP ────────────────────────────────────────────────────────────────
    rows = []

    for perc in PERCENTS:
        print(f"Processing {perc}% data...")
        pkl_path = DATA_DIR / f"{perc}_percent_stats_20250623.pkl"
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        for sample_id, stat in enumerate(loaded['img_stats']):
            img         = df.image_preprocess(stat['image'])
            reg_points  = stat['regular_points']
            def_points  = stat['defect_points']

            cell, origin, length = df.estimate_hex_lattice(
                points := np.vstack((reg_points, def_points))
            )
            perf_dots, perf_defects, rot_ang, length = layers.make_perfect_dots(
                reg_points, def_points, cell, img, plot=False
            )
            blocks = layers.find_layer_new(
                perf_dots, perf_defects, rot_ang, med_len_x=length, plot=False, verbose=False,  phase_correct=True
            )
            if blocks is None:
                print(f"Skipping {perc}% sample {sample_id} due to no blocks found.")
                continue  # skip bad images

            # add vacancy-size info
            vac_df = pd.DataFrame(perf_defects, columns=['x', 'y'])
            vac_df['vac_size'] = [COLOR2SIZE[c] for c in stat['colors']]
            blocks[['x', 'y']] = blocks[['x', 'y']].astype(float).round(3)
            vac_df[['x', 'y']] = vac_df[['x', 'y']].round(3)
            blocks = blocks.merge(vac_df, on=['x', 'y'], how='left')

            # classification + connectivity
            blocks = classify_vacancies(blocks, img=img)

            # persist

            blocks['percent']   = perc
            blocks['sample_id'] = sample_id

            blocks['cluster_uid'] = (blocks['percent'].astype(str) + '_' + blocks['sample_id'].astype(str) +
                        '_' +
                        blocks['cluster_id'].astype(str))

            rows.append(blocks[['percent', 'sample_id',
                                'group', 'label', 'vac_size',
                                'category', 'cluster_uid',  # new column
                                'x', 'y']])

    df_all = pd.concat(rows, ignore_index=True)
    return df_all


def save_df(df_all, file_name = "data"):
    df_all.to_csv(f"{file_name}.csv", index=False)


def load_df(file_name = "data"):
    df_all = pd.read_csv(f"{file_name}.csv")
    return df_all

def get_all_blocks(MC_data):
    pos   = MC_data.atoms.pos[:, :3]
    atype = MC_data.atoms.atype

    z_vals = pos[(atype == 1) | (atype == 5), 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    layer_centers = np.linspace(z_min, z_max, 3)

    def assign_group(z):
        i = np.argmin(np.abs(layer_centers - z)) 
        return (2, 1, 3)[i]

    label_map = {
        1: 'atom',   # Ti
        2: 'C',
        3: 'C',
        4: 'O',
        5: 'defect',
        6: 'vC',
        7: 'ST'
    }

    df = pd.DataFrame(pos, columns=['x', 'y', 'z'])
    df['atype'] = atype
    df['label'] = pd.Series(atype).map(label_map).fillna('unknown')
    df['group'] = -1
    mask = (atype == 1) | (atype == 5)
    df.loc[mask, 'group'] = df.loc[mask, 'z'].apply(assign_group)

    return df



import os
import pandas as pd
import atomman as am  # assuming you already import this elsewhere

def make_modeling_df_NEW(data_dir="NEW_data_30nm", include_random=False):
    dfs = []

    for folder, _, files in os.walk(data_dir):
        folder_name = os.path.basename(folder)

        # ── skip non-data folders (like .DS_Store, data_30nm_3.5Ti, etc.) ──
        if not (folder_name.startswith("O") and "_F" in folder_name):
            continue

        folder_parts = folder_name.split("_")
        try:
            ox = float(folder_parts[0][1:])  # after "O"
            fl = float(folder_parts[1][1:])  # after "F"
        except (IndexError, ValueError):
            print(f"⚠️ Skipping folder {folder_name} (unexpected name format)")
            continue

        size = 30
        Tx = ox + fl

        for fname in files:
            # ── skip irrelevant files unless include_random ──
            if not include_random:
                if not fname.endswith(".out.100000"):
                    continue
                run_type = "Relaxed"
            else:
                run_type = "Random"
                if fname.endswith(".out.100000"):
                    run_type = "Relaxed"

            load_path = os.path.join(folder, fname)

            # ── parse file name metadata ──
            try:
                vTi_str = fname.split("_")[0][4:]
                vC_str  = fname.split("_")[1][3:]
                vTi = float(vTi_str)
                vC  = float(vC_str)
            except Exception:
                print(f"⚠️ Skipping file {fname} (unexpected filename format)")
                continue

            # ── load and classify ──
            MC_data = am.load('atom_dump', load_path)
            all_blocks = get_all_blocks(MC_data)
            all_blocks = classify_vacancies(all_blocks)
            all_blocks['category'] = all_blocks['category'].replace(
                {'sandwich': 'complex', 'pit': 'complex'}
            )

            # ── assign metadata ──
            all_blocks['size'] = size
            all_blocks['vTi']  = vTi
            all_blocks['vC']   = vC
            all_blocks['Tx']   = Tx
            if include_random:
                all_blocks['run_type'] = run_type

            dfs.append(all_blocks)

    if not dfs:
        raise ValueError(f"No valid data found in {data_dir}")

    return pd.concat(dfs, ignore_index=True)



def make_modeling_df(conds_30nm = [0, 1, 5, 10], conds_15nm = [
    {'ox': 0,  'fl': 0,  'c': 0},
    {'ox': 0,  'fl': 0,  'c': 1},
    {'ox': 0,  'fl': 0,  'c': 2},
    {'ox': 0,  'fl': 0,  'c': 3},
    {'ox': 60, 'fl': 20, 'c': 0},
    {'ox': 60, 'fl': 20, 'c': 1},
    {'ox': 60, 'fl': 20, 'c': 2},
    {'ox': 60, 'fl': 20, 'c': 3}
]):
    
    dfs = []

    for c in conds_30nm:
        vc = f"0.{c*10:03d}"
        Tx = 0
        size = 30
        vTi = 3.5
        load_path = f"/Users/gguinan/Library/CloudStorage/OneDrive-NREL/spring2025code/for_github2/goldy/data_30nm/O0.000_F0.000/vTi0.035_vC{vc}_CO0.000.out.10000"
        MC_data = am.load('atom_dump', load_path)
        all_blocks = get_all_blocks(MC_data)
        all_blocks = classify_vacancies(all_blocks)
        all_blocks['category'] = all_blocks['category'].replace({'sandwich': 'complex', 'pit': 'complex'})
        all_blocks['size'] = size
        all_blocks['vTi'] = vTi
        all_blocks['vC'] = c
        all_blocks['Tx'] = Tx
        dfs.append(all_blocks)



    for c in conds_15nm:
        ox, fl, vc = c['ox'], c['fl'], c['c']
        vTi = 3
        Tx = ox + fl
        size = 15

        if ox == 0:
            load_path = f'/Users/gguinan/Library/CloudStorage/OneDrive-NREL/spring2025code/for_github2/goldy/runs_0K/O0.{ox}00_F0.{fl}00/vTi0.0{vTi}0_vC0.0{vc}0.out.10000'
        else:
            load_path = f'/Users/gguinan/Library/CloudStorage/OneDrive-NREL/spring2025code/for_github2/goldy/runs_0K/O0.{ox}0_F0.{fl}0/vTi0.0{vTi}0_vC0.0{vc}0.out.10000'

        MC_data = am.load('atom_dump', load_path)
        all_blocks = get_all_blocks(MC_data)
        all_blocks = classify_vacancies(all_blocks)
        all_blocks['category'] = all_blocks['category'].replace({'sandwich': 'complex', 'pit': 'complex'})
        all_blocks['size'] = size
        all_blocks['vTi'] = vTi
        all_blocks['vC'] = vc
        all_blocks['Tx'] = Tx
        dfs.append(all_blocks)


    return pd.concat(dfs, ignore_index=True)


def _make_labels_from_index(idx: pd.MultiIndex) -> pd.Index:
    idx_df = idx.to_frame(index=False)
    labels = (
        "Tx=" + idx_df["Tx"].astype(float).map("{:.1f}".format)
        + ", " + idx_df["size"].astype(int).astype(str) + "nm"
        + ", vTi=" + idx_df["vTi"].astype(float).map("{:.3f}".format)
        + ", vC="  + idx_df["vC"].astype(float).map("{:.3f}".format)
    )
    return pd.Index(labels, name=None)

def _metrics_factory(exp_vec: np.ndarray):
    eps = 1e-12
    def _metrics(run_vec: np.ndarray):
        diff = run_vec - exp_vec
        l1  = float(np.abs(diff).sum())
        tvd = 0.5 * l1
        l2  = float(np.sqrt((diff * diff).sum()))
        cos = 1.0 - float((run_vec @ exp_vec) / (np.linalg.norm(run_vec) * np.linalg.norm(exp_vec) + eps))
        m   = 0.5 * (run_vec + exp_vec)
        jsd = 0.5 * (exp_vec * (np.log2((exp_vec + eps) / (m + eps)))).sum() \
            + 0.5 * (run_vec * (np.log2((run_vec + eps) / (m + eps)))).sum()
        jsd_sqrt = float(np.sqrt(max(jsd, 0.0)))
        chi2 = float(((diff * diff) / (exp_vec + eps)).sum())
        hellinger = float(np.sqrt(((np.sqrt(run_vec) - np.sqrt(exp_vec))**2).sum()) / np.sqrt(2))
        bc = float((np.sqrt(run_vec * exp_vec)).sum())
        bhattacharyya = float(-np.log(bc + eps))
        kl_pq = float((exp_vec * np.log((exp_vec + eps) / (run_vec + eps))).sum())
        kl_qp = float((run_vec * np.log((run_vec + eps) / (exp_vec + eps))).sum())
        return {
            "TVD": tvd, "L1": l1, "L2": l2,
            "CosDist": cos, "√JSD": jsd_sqrt, "Chi2": chi2,
            "Hellinger": hellinger, "Bhattacharyya": bhattacharyya,
            "KL(exp||run)": kl_pq, "KL(run||exp)": kl_qp
        }
    return _metrics

def _insert_gap_rows(df: pd.DataFrame, order_idx, cats):
    """Ensure gap labels exist as zero rows so plotting never breaks."""
    out = df.copy()
    for g in ["_gap1_", "_gap2_", "_gap3_"]:
        if g in order_idx and g not in out.index:
            out.loc[g] = 0.0
    # keep column order and reindex to requested order
    out = out.reindex(order_idx)
    # fill any NaNs produced by gap rows
    return out.fillna(0.0)[cats]

def build_props(
    all_blocks_model_relaxed,
    all_blocks_model_random,
    experimental,
    cats
):
    """
    Returns:
        props (DataFrame): proportion table (rows=runs incl. Experiment/Random; cols=categories)
        tx_groups (dict): {'tx0': [...], 'tx05': [...], 'tx08': [...]}
        rand_scores (dict): metrics for the Random row vs Experiment
        metrics_df_raw (DataFrame): per-run L1 & TVD vs Experiment
    """
    # counts for RELAXED runs
    model = all_blocks_model_relaxed[["category", "size", "vTi", "vC", "Tx"]].copy()
    counts_runs = (
        model.groupby(["Tx", "size", "vTi", "vC", "category"]).size()
             .unstack("category", fill_value=0)
             .reindex(columns=cats, fill_value=0)
    )
    counts_runs = counts_runs.sort_index(level=["Tx", "vC", "size", "vTi"], ascending=True)

    # label rows
    labels = _make_labels_from_index(counts_runs.index)
    counts_runs.index = labels

    # add one "Random" group: pick first group for all_blocks_model_random
    grp = all_blocks_model_random.groupby(["Tx", "size", "vTi", "vC"], sort=True)
    pick_key = sorted(grp.groups.keys())[0]
    df_one = grp.get_group(pick_key)
    counts_one = df_one["category"].value_counts().reindex(cats, fill_value=0)
    counts_runs = pd.concat([counts_runs, pd.DataFrame([counts_one], index=["Random"])])

    # add Experiment
    exp_row = experimental["category"].value_counts().reindex(cats, fill_value=0)
    counts_runs = pd.concat([counts_runs, pd.DataFrame([exp_row], index=["Experiment"])])

    # to proportions
    props = counts_runs.div(counts_runs.sum(axis=1).replace(0, 1), axis=0)

    # split by Tx
    tx0  = [s for s in labels if "Tx=0.0" in s]
    tx05 = [s for s in labels if "Tx=0.5" in s]
    tx08 = [s for s in labels if "Tx=0.8" in s]

    order_idx = ["Experiment", "Random", "_gap1_"] + tx0 + ["_gap2_"] + tx05 + ["_gap3_"] + tx08
    props = _insert_gap_rows(props, order_idx, cats)

    # metrics vs Experiment
    exp_vec = props.loc["Experiment"].to_numpy()
    metrics_fn = _metrics_factory(exp_vec)
    rand_scores = metrics_fn(props.loc["Random"].to_numpy())

    gap_labels = {"_gap1_", "_gap2_", "_gap3_"}
    labels_to_eval = [lab for lab in props.index if lab not in gap_labels]

    rows = []
    for lab in labels_to_eval:
        vec = props.loc[lab].to_numpy()
        m = metrics_fn(vec)
        rows.append({"Label": lab, "L1": m["L1"], "TVD": m["TVD"]})
    metrics_df_raw = pd.DataFrame(rows)

    tx_groups = {"tx0": tx0, "tx05": tx05, "tx08": tx08}
    return props, tx_groups, rand_scores, metrics_df_raw

def plot_three_panel(
    props: pd.DataFrame,
    cats: list[str],
    cat_palette: dict[str, str],
    tx_groups: dict[str, list[str]],
    rand_scores: dict,
    ylim: tuple[float, float] = (0, 0.75),
    figsize=(20, 5),
):
    """
    Plots: (top) stacked category proportions; (bottom) TVD vs Experiment for each Tx group.
    Returns: fig, {'top': ax_top, 'tx0': ax0, 'tx05': ax05, 'tx08': ax08}
    """
    # figure & top bar
    N = len(props.index)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows=2, ncols=N, height_ratios=[3, 3])

    ax_top = fig.add_subplot(gs[0, :])
    props.plot(kind="bar", stacked=True, ax=ax_top,
               color=[cat_palette[c] for c in cats],
               width=0.85, edgecolor="none")
    ax_top.set_xlabel("")
    ax_top.set_xticks([])
    ax_top.set_xticklabels([])
    if ax_top.get_legend():
        ax_top.get_legend().remove()
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # spans for bottom axes
    pos = {lab: i for i, lab in enumerate(props.index)}
    def _span(labels):
        if not labels:
            return slice(0, 0)
        return slice(min(pos[l] for l in labels), max(pos[l] for l in labels) + 1)

    span0  = _span(tx_groups.get("tx0", []))
    span05 = _span(tx_groups.get("tx05", []))
    span08 = _span(tx_groups.get("tx08", []))

    ax0  = fig.add_subplot(gs[1, span0])
    ax05 = fig.add_subplot(gs[1, span05], sharey=ax0)
    ax08 = fig.add_subplot(gs[1, span08], sharey=ax0)

    # metrics plotter
    exp_vec = props.loc["Experiment"].to_numpy()
    metrics_fn = _metrics_factory(exp_vec)
    gap_labels = {"_gap1_", "_gap2_", "_gap3_"}

    def _plot_metrics(ax, run_labels, title=None, show_ylabel=False):
        run_labels = [r for r in run_labels if r not in gap_labels]
        x = np.arange(len(run_labels))
        series = [metrics_fn(props.loc[r].to_numpy())["TVD"] for r in run_labels]
        rand_val = rand_scores["TVD"]

        ax.plot(x, series, marker="o", alpha=0.8)
        ax.axhline(rand_val, linestyle="--", color="black")
        ax.set_xlim(-0.25, len(run_labels) - 0.75 if len(run_labels) else 0.75)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        if title:
            ax.set_title(title)
        if not show_ylabel:
            ax.tick_params(labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _plot_metrics(ax0,  tx_groups.get("tx0", []),  title=None, show_ylabel=True)
    _plot_metrics(ax05, tx_groups.get("tx05", []), title=None, show_ylabel=False)
    _plot_metrics(ax08, tx_groups.get("tx08", []), title=None, show_ylabel=False)

    axes = {"top": ax_top, "tx0": ax0, "tx05": ax05, "tx08": ax08}
    return fig, axes
