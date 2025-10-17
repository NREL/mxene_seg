import random
import numpy as np
import atomap.api as am
from PIL import Image
import matplotlib.pyplot as plt
import atomai as aoi
import pandas as pd 
from matplotlib_scalebar.scalebar import ScaleBar
import hyperspy.api as hs
from matplotlib.colors import ListedColormap
import cv2
from skimage.segmentation import flood_fill
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from scipy import spatial, ndimage, optimize
from scipy.spatial import cKDTree
from matplotlib.lines import Line2D
from atomai.models.segmentor import Segmentor
import torch
import os
from contextlib import contextmanager
import sys

### ================================
### Major Functions
### ================================

# - create_training_data: generates training image patches and corresponding atom/mask labels by cropping the input image and using 2D Gaussian fitting; 
#                         supports optional adatom detection and a range of crop sizes
# - get_total_points:     Extracts and filters atomic coordinates from an image using a neural network model across multiple threshold levels, applying 
#                         spatial filtering to avoid duplicate detections.
 

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_atomai_model(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta_dict = torch.load(path, map_location=device, weights_only=False)
    model_type = meta_dict.pop("model_type")

    if model_type == "seg":
        model_name = meta_dict.pop("model")
        nb_classes = meta_dict.pop("nb_classes")
        weights = meta_dict.pop("weights")
        model = Segmentor(model_name, nb_classes, **meta_dict)
        model.net.load_state_dict(weights)
        if "optimizer" in meta_dict.keys():
            optimizer = meta_dict.pop("optimizer")
            model.optimizer = optimizer
        model.net.eval()
    return model

def create_circular_mask(image_shape, centers, r=6):
    mask = np.zeros(image_shape, dtype=np.uint8)
    yy, xx = np.ogrid[:image_shape[0], :image_shape[1]]
    for cx, cy in centers:
        cx, cy = int(round(cx)), int(round(cy))
        distance = (yy - cy)**2 + (xx - cx)**2
        mask[distance <= r**2] = 1
    return mask


def build_image_and_mask_lists(
        crops, model, df,
        *,
        r=5,
        size=(256, 256),
        t_values=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2),
        max_iter=15,
        grid_size=150,
        margin=0.05,
        progress_every=50,
):
    images, masks = [], []
    for i, img in enumerate(crops):
        with suppress_stdout():
            img_arr, _, tpoints = get_total_points(
                model, img, size=size, plot=False, plot2=False, t_values=t_values
            )

        defects = df.iterative_atom_placement(
            tpoints, max_iter=max_iter, grid_size=grid_size,
            margin=margin, verbose=False, plot=False
        )
        defects = [
            np.asarray(d) for d in defects
            if isinstance(d, (list, np.ndarray)) and np.asarray(d).ndim == 2 and np.asarray(d).shape[1] == 2
        ]
        defects = np.vstack(defects) if defects else np.empty((0, 2))

        mask = (
            create_circular_mask(img_arr.shape, defects, r=r)
            if len(defects) else np.zeros_like(img_arr)
        )

        images.append(img_arr)
        masks.append(mask)

        if progress_every and i % progress_every == 0:
            print(f"image {i}")

    return images, masks





def build_image_and_mask_lists_LATTICE(
        crops, model, df,
        *,
        r=5,
        size=(256, 256),
        t_values=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2),
        max_iter=15,
        grid_size=150,
        margin=0.05,
        progress_every=50,
):
    images, masks = [], []
    for i, img in enumerate(crops):
        with suppress_stdout():
            img_arr, _, tpoints = get_total_points(
                model, img, size=size, plot=False, plot2=False, t_values=t_values
            )

        defects = df.iterative_atom_placement(
            tpoints, max_iter=max_iter, grid_size=grid_size,
            margin=margin, verbose=False, plot=False
        )
        defects = [
            np.asarray(d) for d in defects
            if isinstance(d, (list, np.ndarray)) and np.asarray(d).ndim == 2 and np.asarray(d).shape[1] == 2
        ]
        defects = np.vstack(defects) if defects else np.empty((0, 2))
        tpoints = np.asarray(tpoints).reshape(-1, 2)        # ensure 2-column array
        combined = np.vstack([tpoints, defects]) if defects.size else tpoints
        mask = (
            create_circular_mask(img_arr.shape, combined, r=r)
            if len(defects) else np.zeros_like(img_arr)
        )

        images.append(img_arr)
        masks.append(mask)

        if progress_every and i % progress_every == 0:
            print(f"image {i}")

    return images, masks

def create_training_data(image, n_samples, min_distance, crop_dim=256, size_range=None,
                         scale=7, radius_mask=5, adatom=False, adatom_ratio=1.0,
                         highlight_radius=8, highlight_count=1, single_class=True, plot=True):
    """
    Generates training data by randomly cropping patches from an input image and creating 
    corresponding atom position masks using 2D Gaussian fitting.

    Parameters:
        image (PIL.Image or array): Input image.
        n_samples (int): Number of training samples (patches) to generate.
        min_distance (float): Minimum distance between detected atomic positions.
        crop_dim (int): Target dimension for cropped patches (square). Default is 256.
        size_range (tuple): If given, crops will have random size within this range.
        scale (int): Scaling factor for mask creation.
        radius_mask (int): Radius for individual atom mask spots.
        adatom (bool): Whether to edit training data to fix labels around adatoms and include more adatom crops.
        highlight_radius (int): Radius of circles drawn around brightest spots (for adatom mode).
        highlight_count (int): Number of bright spots to highlight per image (for adatom mode).
        single_class (bool): If False, separate class masks for adatoms will be included.
        plot (bool): Whether to plot sample images, positions, and masks.

    Returns:
        images (ndarray): Array of cropped image patches, shape (n_samples, 1, crop_dim, crop_dim).
        labels (ndarray): Corresponding label masks, shape (n_samples, 1, crop_dim, crop_dim).
        coords (list of DataFrames): Atomic coordinates for each sample.
    """

    if adatom:
        circled_image, original_image, selected_adatoms = circle_brightest_spot(np.array(image), highlight_radius, plot=plot, count=highlight_count)

        n_adatom = int(n_samples * adatom_ratio)
        n_random = n_samples - n_adatom

        if size_range is None:
            circled_crops, base_crops, circled_signals, base_signals = crop_random_patches(
                [circled_image, original_image], iter=n_adatom, fixed_size=crop_dim, adatom=True)
            rand_crops, rand_signals = crop_random_patches(
                original_image, iter=n_random, fixed_size=crop_dim, adatom=False)
        else:
            circled_crops, base_crops, circled_signals, base_signals = crop_random_patches(
                [circled_image, original_image], iter=n_adatom, size_range=size_range, adatom=True, adatoms = selected_adatoms)
            rand_crops, rand_signals = crop_random_patches(
                original_image, iter=n_random, size_range=size_range, adatom=False)

        base_crops.extend(rand_crops)
        base_signals.extend(rand_signals)

        circled_coords = get_atom_coords(circled_signals, distance=min_distance)
        base_coords = get_atom_coords(base_signals, distance=min_distance)

        merged_coords, adatom_coords = [], []
        for c_df, b_df in zip(circled_coords, base_coords[:n_adatom]):
            df_all = pd.merge(c_df, b_df, on=['x', 'y'], how='outer')
            df_unique = pd.merge(b_df, c_df, on=['x', 'y'], how='left', indicator=True)
            unique = df_unique[df_unique['_merge'] == 'left_only'].drop(
                columns=[col for col in c_df.columns if col not in b_df.columns])
            merged_coords.append(df_all)
            adatom_coords.append(unique)

        merged_coords.extend(base_coords[n_adatom:])
        adatom_coords.extend([pd.DataFrame(columns=['x', 'y'])] * n_random)

        data = get_masks(base_signals, merged_coords, scale, radius_mask)
        if not single_class:
            data = edit_masks(data, adatom_coords)

        labels = [arr.astype(float) for arr in data['labels']] if size_range is None else data['labels']
        images = base_crops

    else:
        if size_range is None:
            images, signals = crop_random_patches(image, iter=n_samples, fixed_size=crop_dim)
        else:
            images, signals = crop_random_patches(image, iter=n_samples, size_range=size_range)

        coords = get_atom_coords(signals, distance=min_distance)
        data = get_masks(images, coords, scale, radius_mask)
        labels = data['labels']

    if size_range is not None:
        coords = merged_coords if adatom else coords
        images, coords, labels = resizing(images, crop_dim, coords, labels)

    images = np.array(images).reshape(n_samples, 1, crop_dim, crop_dim)
    labels = np.array(labels).reshape(n_samples, 1, crop_dim, crop_dim)

    if plot:
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(5):
            ax[i].imshow(np.array(images[i][0]), cmap="gray")
            ax[i].axis('off')
            ax[i].set_title(f"Image {i}")
        plt.subplots_adjust(top=0.85)
        plt.show()

        fig = plt.figure(figsize=(15, 6))
        for i in range(5):
            ax1 = fig.add_subplot(2, 5, i + 1)
            ax1.imshow(np.array(images[i][0]), cmap="gray")
            ax1.scatter(coords[i]['x'], coords[i]['y'], s=15, c="red")
            ax1.set_title(f"Positions {i}")
            ax1.axis('off')

            ax2 = fig.add_subplot(2, 5, i + 6)
            ax2.imshow(labels[i][0])
            ax2.set_title(f"Mask {i}")
            ax2.axis('off')
        plt.subplots_adjust(top=0.85)
        plt.show()
    
    coords = merged_coords if adatom else coords
    return images, labels, coords

def get_mask(model, image, size = [512, 512], plot = False):
    w, h = image.size  
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    image = image.crop((left, top, right, bottom))

    if image.mode == "RGBA":
        test_image2 = image.resize(size, Image.LANCZOS).convert("L")  
    else:
        test_image2 = image.resize(size, Image.LANCZOS)
    img_arr = np.array(test_image2)

    nn_output, coordinates = model.predict(img_arr, thresh = .5)
    mask = nn_output.squeeze()
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)

        axes[0].imshow(img_arr, cmap='gray')
        axes[0].set_title('image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('mask')
        axes[1].axis('off')

    
    return mask



def get_total_points(model, image, plot = False, plot2 = True, t_values = None, size = [512, 512], single_class = True, dist = .6):
    """
    Extracts and filters atomic coordinates from an image using a neural network model across 
    multiple threshold levels, applying spatial filtering to avoid duplicate detections.

    Parameters:
        model (object): A model with a `.predict(image, thresh)` method that returns a 
            segmentation mask and predicted coordinates.
        image (PIL.Image): Input image to process.
        plot (bool): If True, displays a plot of filtered points overlaid on the mask 
            for each threshold. Default is False.
        plot2 (bool): If True, shows the final set of detected points over the original image. 
            Default is True.
        t_values (list of float): List of threshold values for progressively filtering the 
            segmentation mask. If None, a default list is used.
        size (list of int): Target size [width, height] for resizing the input image before 
            processing. Default is [512, 512].

    Returns:
        img_arr (ndarray): Resized grayscale image array passed into the model.
        mask (ndarray): Segmentation mask produced by the model.
        stored_points (ndarray): Final array of filtered atomic coordinates, shape (N, 2).
    """

    if t_values is None:
        t_values = sorted([0.9, 0.85, .8, 0.75, 0.7, 0.65, 0.6, .55, .5, .45, .4], reverse=True)
    
    cmap = cm.get_cmap("tab10", len(t_values))  # "autumn" colormap for a red-yellow gradient
    color_map = [mcolors.to_hex(cmap(i)) for i in range(len(t_values))]  # Convert to hex colors

    stored_points = None  # Store already plotted points
    trees = []  # Store KD trees for distance queries

    w, h = image.size  
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    image = image.crop((left, top, right, bottom))


    if image.mode == "RGBA":
        test_image2 = image.resize(size, Image.LANCZOS).convert("L")  
    else:
        test_image2 = image.resize(size, Image.LANCZOS)
    img_arr = np.array(test_image2)

    nn_output, coordinates = model.predict(img_arr, thresh = .5)
    y, x, c = coordinates[0].T
    mask = nn_output.squeeze()
    med = 0
    for i, t in enumerate(t_values):
        new_thresh_img = mask > t
        labels, nlabels = ndimage.label(new_thresh_img)
        coordinates = np.array(ndimage.center_of_mass(new_thresh_img, labels, np.arange(nlabels) + 1))
        coordinates = coordinates.reshape(-1, 2)[:, :2] if coordinates.size else np.empty((0, 2))
        y, x = coordinates.T if coordinates.size else (np.empty(0), np.empty(0))
        points = (np.column_stack((x, y))  if x.size else np.empty((0, 2))
)


        if len(points) == 0:
            continue
        if med == 0:
            med = med_dist(points)
            if plot:
                plt.imshow(mask, cmap="gray", origin = "lower")
                plt.axis("off")
                plt.scatter(points[:, 0], points[:, 1], s=3, c=color_map[i], label=f"t={t}")
            filtered_points = points
        
        tree = cKDTree(points)
        filter_radius = dist * med
        pairs = tree.query_pairs(r=filter_radius)  # Find close pairs
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        G.add_edges_from(pairs)
        selected_indices = [list(cluster)[0] for cluster in nx.connected_components(G)]
        points = points[selected_indices]
        tree = cKDTree(points)

        if i >0:
            distances = np.array([tree.query(points, k=1)[0] for tree in trees])
            filtered_points = points[np.all(distances > (dist*med), axis=0)]
            if plot:
                plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=5, c=color_map[i], label=f"t={t}")

        # Store points and KDTree for the next iterations
        if stored_points is None:
            stored_points = filtered_points
        elif filtered_points.shape[0]>0:
            stored_points = np.vstack((stored_points, filtered_points))

        trees.append(cKDTree(points))

    if plot:
        plt.legend(fontsize=8, markerscale=0.5, loc="upper right", framealpha=1)
        plt.show()

    if plot2 and single_class:
        plt.imshow(img_arr, cmap="gray", origin = "lower")
        plt.scatter(stored_points[:,0], stored_points[:,1], s = 5, c = "red")
        plt.axis('off')
        plt.show()

    elif plot2 and not single_class:
        x = stored_points[:, 0].astype(int)
        y = stored_points[:, 1].astype(int)

        red = mask[y, x, 0]
        green = mask[y, x, 1]
        blue = mask[y, x, 2]

        # Class 1 if green is dominant, else 0
        indices = (green > red) & (green > blue)
        indices = indices.astype(int)
        sizes = np.where(indices == 1, 20, 3)  # green: 10, red: 3


        legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='regular atom',
            markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='adatom',
            markerfacecolor='mediumblue', markersize=8)
    ]

        cmap = ListedColormap(['red', 'mediumblue'])
        plt.imshow(img_arr, cmap="gray", origin="lower")
        plt.scatter(stored_points[:,0], stored_points[:,1], s=sizes, c=indices, cmap=cmap, vmin=0, vmax=1)
        plt.legend(handles=legend_elements, loc='upper left')
        plt.axis('off')
        plt.show()

        stored_colors = indices.reshape(-1, 1)
        stored_points = np.hstack((stored_points, stored_colors))


    return img_arr, mask, stored_points







def get_total_points_ensemble(model, image, plot = False, plot2 = True, t_values = None, size = [512, 512], single_class = True, dist = .6, normalize = False):
    if t_values is None:
        t_values = sorted([0.9, 0.85, .8, 0.75, 0.7, 0.65, 0.6, .55, .5, .45, .4], reverse=True)
    
    cmap = cm.get_cmap("tab10", len(t_values))  # "autumn" colormap for a red-yellow gradient
    color_map = [mcolors.to_hex(cmap(i)) for i in range(len(t_values))]  # Convert to hex colors

    stored_points = None  # Store already plotted points
    trees = []  # Store KD trees for distance queries

    w, h = image.size  
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    image = image.crop((left, top, right, bottom))


    if image.mode == "RGBA":
        test_image2 = image.resize(size, Image.LANCZOS).convert("L")  
    else:
        test_image2 = image.resize(size, Image.LANCZOS)
    img_arr = np.array(test_image2)

    nn_out_mean, nn_out_var = model.predict(img_arr)
    #nn_out_mean = (nn_out_mean - nn_out_mean.min()) / (nn_out_mean.max() - nn_out_mean.min())
    if normalize:
        nn_out_mean = normalize_with_clipping(nn_out_mean, lower_percentile=10, upper_percentile=95)
    loc = aoi.predictors.predictor.Locator(threshold= .5)
    coordinates = loc.run(nn_out_mean, image)
    y, x, c = coordinates[0].T
    mask = nn_out_mean.squeeze()

    for i, t in enumerate(t_values):
        new_thresh_img = mask > t
        labels, nlabels = ndimage.label(new_thresh_img)
        coordinates = np.array(ndimage.center_of_mass(new_thresh_img, labels, np.arange(nlabels) + 1))
        coordinates = coordinates.reshape(-1, 2)[:, :2] if coordinates.size else np.empty((0, 2))
        y, x = coordinates.T if coordinates.size else (np.empty(0), np.empty(0))
        points = np.array(list(zip(x, y)))

        if i == 0:
            med = med_dist(points)
            if plot:
                plt.imshow(mask, cmap="gray", origin = "lower")
                plt.axis("off")
                plt.scatter(points[:, 0], points[:, 1], s=3, c=color_map[i], label=f"t={t}")
            filtered_points = points
        
        tree = cKDTree(points)
        filter_radius = dist * med
        pairs = tree.query_pairs(r=filter_radius)  # Find close pairs
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        G.add_edges_from(pairs)
        selected_indices = [list(cluster)[0] for cluster in nx.connected_components(G)]
        points = points[selected_indices]
        tree = cKDTree(points)

        if i >0:
            distances = np.array([tree.query(points, k=1)[0] for tree in trees])
            filtered_points = points[np.all(distances > (dist*med), axis=0)]
            if plot:
                plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=5, c=color_map[i], label=f"t={t}")

        # Store points and KDTree for the next iterations
        if stored_points is None:
            stored_points = filtered_points
        elif filtered_points.shape[0]>0:
            stored_points = np.vstack((stored_points, filtered_points))

        trees.append(cKDTree(points))

    if plot:
        plt.legend(fontsize=8, markerscale=0.5, loc="upper right", framealpha=1)
        plt.show()

    if plot2 and single_class:
        plt.imshow(img_arr, cmap="gray", origin = "lower")
        plt.scatter(stored_points[:,0], stored_points[:,1], s = 5, c = "red")
        plt.axis('off')
        plt.show()

    elif plot2 and not single_class:
        x = stored_points[:, 0].astype(int)
        y = stored_points[:, 1].astype(int)

        red = mask[y, x, 0]
        green = mask[y, x, 1]
        blue = mask[y, x, 2]

        # Class 1 if green is dominant, else 0
        indices = (green > red) & (green > blue)
        indices = indices.astype(int)
        sizes = np.where(indices == 1, 20, 3)  # green: 10, red: 3


        legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='regular atom',
            markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='adatom',
            markerfacecolor='mediumblue', markersize=8)
    ]

        cmap = ListedColormap(['red', 'mediumblue'])
        plt.imshow(img_arr, cmap="gray", origin="lower")
        plt.scatter(stored_points[:,0], stored_points[:,1], s=sizes, c=indices, cmap=cmap, vmin=0, vmax=1)
        plt.legend(handles=legend_elements, loc='upper left')
        plt.axis('off')
        plt.show()

        stored_colors = indices.reshape(-1, 1)
        stored_points = np.hstack((stored_points, stored_colors))

    return img_arr, mask, stored_points, nn_out_var.squeeze()


def load_model(path, *, device=None, eval_mode=True):
    """
    Returns either a trained Segmentor or an EnsemblePredictor
    depending on the checkpoint contents.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # ── single-network Segmentor checkpoint (.tar) ────────────────────────────
    if 'model_type' in ckpt:                            # AtomAI .tar
        if ckpt['model_type'] != 'seg':
            raise ValueError(f"Unsupported model_type: {ckpt['model_type']}")

        model = Segmentor(
            ckpt['model'],
            ckpt['nb_classes'],
            **{k: v for k, v in ckpt.items()
               if k not in ('model_type', 'model', 'nb_classes', 'weights',
                             'optimizer')}
        )
        model.net.load_state_dict(ckpt['weights'])
        if 'optimizer' in ckpt:
            model.optimizer = ckpt['optimizer']
        if eval_mode:
            model.net.eval()
        return model

    # ── ensemble predictor checkpoint (.pt) ───────────────────────────────────
    if {'smodel', 'ensemble'} <= ckpt.keys():           # ensemble .pt
        return aoi.predictors.EnsemblePredictor(
            ckpt['smodel'], ckpt['ensemble']
        )

    raise ValueError("Unrecognised checkpoint format")


def get_total_points_mask( image, mask, plot = False, plot2 = True, t_values = None, size = [512, 512], single_class = True, dist = .6, normalize = False):

    if t_values is None:
        t_values = sorted([0.9, 0.85, .8, 0.75, 0.7, 0.65, 0.6, .55, .5, .45, .4], reverse=True)
    
    cmap = cm.get_cmap("tab10", len(t_values))  # "autumn" colormap for a red-yellow gradient
    color_map = [mcolors.to_hex(cmap(i)) for i in range(len(t_values))]  # Convert to hex colors

    stored_points = None  # Store already plotted points
    trees = []  # Store KD trees for distance queries

    #loc = aoi.predictors.predictor.Locator(threshold= .5)
    #coordinates = loc.run(np.expand_dims(np.array(mask), axis=-1), np.array(image))
    #y, x, c = coordinates[0].T

    for i, t in enumerate(t_values):
        new_thresh_img = mask > t
        labels, nlabels = ndimage.label(new_thresh_img)
        coordinates = np.array(ndimage.center_of_mass(new_thresh_img, labels, np.arange(nlabels) + 1))
        coordinates = coordinates.reshape(-1, 2)[:, :2] if coordinates.size else np.empty((0, 2))
        y, x = coordinates.T if coordinates.size else (np.empty(0), np.empty(0))
        points = np.array(list(zip(x, y)))

        if i == 0:
            med = med_dist(points)
            if plot:
                plt.imshow(mask, cmap="gray", origin = "lower")
                plt.axis("off")
                plt.scatter(points[:, 0], points[:, 1], s=3, c=color_map[i], label=f"t={t}")
            filtered_points = points
        
        tree = cKDTree(points)
        filter_radius = dist * med
        pairs = tree.query_pairs(r=filter_radius)  # Find close pairs
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        G.add_edges_from(pairs)
        selected_indices = [list(cluster)[0] for cluster in nx.connected_components(G)]
        points = points[selected_indices]
        tree = cKDTree(points)

        if i >0:
            distances = np.array([tree.query(points, k=1)[0] for tree in trees])
            filtered_points = points[np.all(distances > (dist*med), axis=0)]
            if plot:
                plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=5, c=color_map[i], label=f"t={t}")

        # Store points and KDTree for the next iterations
        if stored_points is None:
            stored_points = filtered_points
        elif filtered_points.shape[0]>0:
            stored_points = np.vstack((stored_points, filtered_points))

        trees.append(cKDTree(points))

    if plot:
        plt.legend(fontsize=8, markerscale=0.5, loc="upper right", framealpha=1)
        plt.show()

    if plot2 and single_class:
        plt.imshow(np.array(image), cmap="gray", origin = "lower")
        plt.scatter(stored_points[:,0], stored_points[:,1], s = 5, c = "red")
        plt.axis('off')
        plt.show()

    elif plot2 and not single_class:
        x = stored_points[:, 0].astype(int)
        y = stored_points[:, 1].astype(int)

        red = mask[y, x, 0]
        green = mask[y, x, 1]
        blue = mask[y, x, 2]

        # Class 1 if green is dominant, else 0
        indices = (green > red) & (green > blue)
        indices = indices.astype(int)
        sizes = np.where(indices == 1, 20, 3)  # green: 10, red: 3


        legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='regular atom',
            markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='adatom',
            markerfacecolor='mediumblue', markersize=8)
    ]

        cmap = ListedColormap(['red', 'mediumblue'])
        plt.imshow(np.array(image), cmap="gray", origin="lower")
        plt.scatter(stored_points[:,0], stored_points[:,1], s=sizes, c=indices, cmap=cmap, vmin=0, vmax=1)
        plt.legend(handles=legend_elements, loc='upper left')
        plt.axis('off')
        plt.show()

        stored_colors = indices.reshape(-1, 1)
        stored_points = np.hstack((stored_points, stored_colors))







    return stored_points



### ================================
###  Helper Functions
### ================================

# - med_dist: calculates the median distance to the second-nearest neighbor for a set of points
# - edit_masks: modifies label masks to distinguish background, atoms, and adatoms
# - get_atom_coords: extracts atom coordinates from AtomAI signals
# - get_masks: generates masks using coordinates and AtomAI utilities
# - resizing: resizes images, masks, and updates atom coordinates accordingly
# - circle_brightest_spot: highlights brightest spots in image with filled circles
# - crop_random_patches: randomly crops image patches (with or without adatom overlay)


def med_dist(points):
    """
    Computes the median distance to the second-nearest neighbor for a set of 2D points.

    Parameters:
        points (ndarray): Array of shape (N, 2) representing point coordinates.

    Returns:
        float: Median distance to the second-nearest neighbor.
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    return np.median(distances[:, 1])


def edit_masks(data, adatom_dfs):
    """
    Edits label masks to reassign background, atoms, and adatoms for training.

    Parameters:
        data (dict): Dictionary containing 'labels' key with mask arrays.
        adatom_dfs (list of DataFrames): Adatom coordinates for each mask.

    Returns:
        dict: Modified data with updated 'labels' masks.
    """
    masks = data['labels']
    for i, mask in enumerate(masks):
        mask[mask == 0] = 2
        mask[mask == 1] = 0

        df = adatom_dfs[i]
        if df.empty:
            continue

        for _, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            mask = flood_fill(mask, (y, x), 1, connectivity=2)

        data['labels'][i] = mask
    return data


def get_atom_coords(signal_list, distance):
    """
    Extracts atomic coordinates from a list of AtomAI signal objects.

    Parameters:
        signal_list (list): List of AtomAI Signal2D objects.
        distance (float): Minimum distance for atom detection.

    Returns:
        list of DataFrames: Extracted (x, y) coordinates for each signal.
    """
    dfs = []
    for signal in signal_list:
        coords = am.get_atom_positions(signal, distance)
        dfs.append(pd.DataFrame(coords, columns=['x', 'y']))
    return dfs


def get_masks(images, dfs, scale, radius):
    """
    Generates binary atom masks from coordinates using AtomAI utilities.

    Parameters:
        images (list): List of image arrays.
        dfs (list of DataFrames): Atom coordinates per image.
        scale (int or list): Scaling factor(s) for mask creation.
        radius (int or list): Radius/radii for circular atom masks.

    Returns:
        dict: Dictionary with 'images' and 'labels' keys.
    """
    masks = []
    for img, df in zip(images, dfs):
        coords = np.array(df)
        if isinstance(scale, list):
            mask = aoi.utils.create_lattice_mask(img, coords, scale=int(scale.pop(0)), rmask=int(radius.pop(0)))
        else:
            mask = aoi.utils.create_lattice_mask(img, coords, scale=scale, rmask=radius)
        masks.append(mask.T)
    return {'images': images, 'labels': np.array(masks, dtype=object)}


def resizing(images, target_dim, dfs, masks):
    """
    Resizes images and masks to a target dimension, and scales coordinates accordingly.

    Parameters:
        images (list of PIL.Image): Input images.
        target_dim (int): Desired output size (square).
        dfs (list of DataFrames): Atom coordinates for each image.
        masks (list of arrays): Corresponding label masks.

    Returns:
        tuple: (resized_images, adjusted_dfs, resized_masks)
    """
    resized_imgs, resized_masks, adjusted_dfs = [], [], []

    for img, df, mask in zip(images, dfs, masks):
        w_old, h_old = img.size
        img_resized = img.resize((target_dim, target_dim), Image.LANCZOS)
        scale_x, scale_y = target_dim / w_old, target_dim / h_old

        df_scaled = df.copy()
        df_scaled['x'] *= scale_x
        df_scaled['y'] *= scale_y

        mask_resized = Image.fromarray(mask).resize((target_dim, target_dim), Image.NEAREST)

        resized_imgs.append(img_resized)
        adjusted_dfs.append(df_scaled)
        resized_masks.append(np.array(mask_resized))

    return resized_imgs, adjusted_dfs, resized_masks


def circle_brightest_spot(image, radius, count, plot, min_distance=20):
    """
    Highlights the brightest spots in an image by drawing filled circles.

    Parameters:
        image (ndarray): Input grayscale image array.
        radius (int): Radius of circles to draw.
        count (int): Number of spots to highlight.
        plot (bool): If True, display result using matplotlib.
        min_distance (int): Minimum distance between selected spots.

    Returns:
        tuple: (image with circles as PIL, original image as PIL, selected coordinates)
    """
    if count == 0:
        return Image.fromarray(image.copy()), Image.fromarray(image)

    sorted_idx = np.unravel_index(np.argsort(image.ravel())[::-1], image.shape)
    bright_spots = list(zip(sorted_idx[0], sorted_idx[1]))

    circle_overlay = image.copy()
    selected = []

    for y, x in bright_spots:
        if all(np.sqrt((y - sy)**2 + (x - sx)**2) >= min_distance for sy, sx in selected):
            dv = int(np.min(image))
            cv2.circle(circle_overlay, (x, y), radius, color=(dv, dv, dv), thickness=-1)
            selected.append((y, x))
        if len(selected) == count:
            break

    if plot:
        plt.imshow(circle_overlay, cmap='gray')
        plt.axis('off')
        plt.show()

    return Image.fromarray(circle_overlay), Image.fromarray(image), selected


def crop_random_patches(image, iter=10, fixed_size=None, size_range=None,
                        adatom=False, plot=False, adatoms=None):
    """
    Randomly crops square patches from an image (or image pair), optionally centered around adatoms.

    Parameters:
        image (PIL.Image or tuple): Input image or (circled image, base image) tuple for adatom mode.
        iter (int): Number of patches to generate.
        fixed_size (int): If given, all patches will be this size.
        size_range (tuple): If given, patch sizes will be sampled within this range.
        adatom (bool): Whether to sample around known adatom coordinates.
        plot (bool): If True, display cropped patches using matplotlib.
        adatoms (list): List of (y, x) tuples for adatom locations.

    Returns:
        tuple: In adatom mode, returns (circ_crops, base_crops, circ_signals, base_signals).
               Otherwise, returns (crops, signals) or (crops, signals, boxes) if plot=True.
    """
    def normalize_crop(crop):
        norm = (crop - np.min(crop)) / (np.max(crop) - np.min(crop))
        return Image.fromarray((norm * 255).astype(np.uint8))

    if adatom:
        circ_img, base_img = image
        w, h = circ_img.size

        circ_crops, base_crops, circ_signals, base_signals = [], [], [], []

        for _ in range(iter):
            size = fixed_size or round(1 / random.uniform(1 / size_range[1], 1 / size_range[0]))

            if adatoms and len(adatoms) > 0:
                y_atom, x_atom = random.choice(adatoms)
                x_min = max(0, x_atom - size + 1)
                x_max = min(w - size, x_atom)
                y_min = max(0, y_atom - size + 1)
                y_max = min(h - size, y_atom)

                x1 = random.randint(x_min, x_max)
                y1 = random.randint(y_min, y_max)
                x2 = x1 + size
                y2 = y1 + size
            else:
                x1 = random.randint(0, w - size)
                y1 = random.randint(0, h - size)
                x2 = x1 + size
                y2 = y1 + size

            circ_crop = normalize_crop(circ_img.crop((x1, y1, x2, y2)))
            base_crop = normalize_crop(base_img.crop((x1, y1, x2, y2)))

            if plot:
                plt.subplot(1, 2, 1)
                plt.imshow(circ_crop, cmap='gray')
                plt.title('circ_crop')
                plt.subplot(1, 2, 2)
                plt.imshow(base_crop, cmap='gray')
                plt.title('base_crop')
                plt.show()

            circ_crops.append(circ_crop)
            base_crops.append(base_crop)
            circ_signals.append(hs.signals.Signal2D(np.array(circ_crop)))
            base_signals.append(hs.signals.Signal2D(np.array(base_crop)))

        return circ_crops, base_crops, circ_signals, base_signals

    else:
        w, h = image.size
        crops, signals, boxes = [], [], []

        for _ in range(iter):
            size = fixed_size or round(1 / random.uniform(1 / size_range[1], 1 / size_range[0]))
            x1 = random.randint(0, w - size)
            y1 = random.randint(0, h - size)
            x2 = x1 + size
            y2 = y1 + size

            crop = normalize_crop(image.crop((x1, y1, x2, y2)))
            crops.append(crop)
            signals.append(hs.signals.Signal2D(np.array(crop)))
            if plot:
                boxes.append([x1, y1, x2, y2])

        return (crops, signals, boxes) if plot else (crops, signals)




def normalize_with_clipping(arr, lower_percentile=0, upper_percentile=95):
    lower_bound = np.percentile(arr, lower_percentile)
    upper_bound = np.percentile(arr, upper_percentile)
    clipped_arr = np.clip(arr, lower_bound, upper_bound)
    return (clipped_arr - np.min(clipped_arr)) / (np.max(clipped_arr) - np.min(clipped_arr))
