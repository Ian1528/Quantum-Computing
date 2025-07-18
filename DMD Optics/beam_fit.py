from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter 
import math
import os

def find_spots(image, num_peaks, threshold = 50,) -> list[tuple[int, int]]:
    """
    Takes an image and finds the center pixel of the 5x5 array

    Args:
        image (_type_): _description_
        num_peaks (int): number of peaks to find in the image.
        threshold (int, optional): minimum threshold for finding a peak. Defaults to 100.

    Returns:
        list[tuple[int, int]]: list of (x, y) coordinates of the peaks found in the image.
    """
    indices = peak_local_max(image, min_distance = 50, threshold_abs=threshold, exclude_border=0, num_peaks=num_peaks)
    # draw a square around each peak
    if len(indices) != num_peaks:
        raise ValueError(f"{len(indices)} number of peaks found, please adjust the threshold or min_distance parameters.")
    return indices  # Return (x, y) coordinates


def get_2d_spot_indices(spot_indices, array_size=11) -> np.ndarray:
    """
    Converts a 1D array of spot indices into a 2D array of indices, of shape (array_size, array_size, 2).

    Args:
        spot_indices (_type_): 1D array of spot indices, where each index is a tuple (y, x).
        array_size (int, optional): Size of the 1D array. Defaults to 11.

    Returns:
        np.ndarray: 2D array of shape (array_size, array_size, 2), where each element is a tuple (y, x).
    """
    new_indices = np.empty((array_size, array_size, 2))
    indices = spot_indices[np.argsort(spot_indices[:, 0])]
    for i in range(array_size):
        unsorted_temp = indices[i*array_size:(i+1)*array_size]
        new_indices[i] = unsorted_temp[np.argsort(unsorted_temp[:, 1])]

    return new_indices

def draw_boundaries_around_spots_1d(image, array_size, axis=0, plot_info="indices"):
    """
    Draws rectangular boundaries around detected spots in an image and annotates them.
    This function detects spots in the provided image, draws a box around each spot,
    and annotates each box with either the spot's coordinates or its row/column indices.
    The central spot is highlighted in blue, while all others are red.
    Args:
        image (np.ndarray): The input image in which to detect and annotate spots.
        array_size (int): The size of the 1D array of spots (e.g., 11 for a 1x11 grid).
        axis (int, optional): The axis along which to label the indices (0 for x coords, 1 for y coords)
        plot_info (str, optional): Determines the annotation for each spot.
            - "indices": Annotates with row and column indices (default).
            - "coords": Annotates with x and y coordinates.
            - Any other value: No annotation.
    Returns:
        None. Displays the image with annotated spot boundaries using matplotlib.
    """

    spot_indices = find_spots(image, array_size)
    sorted_indices = spot_indices[np.argsort(spot_indices[:, 0])] if axis == 0 else spot_indices[np.argsort(spot_indices[:, 1])]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='viridis')
    box_size = 15  # Half-width of the box (adjust as needed)

    for i in range(len(sorted_indices)):
        spot_y, spot_x = sorted_indices[i]
        edgecolor = "blue" if i == array_size // 2  else "red"
        rect = patches.Rectangle(
        (spot_x - box_size, spot_y - box_size),  # (x, y) of lower left
        2 * box_size, 2 * box_size,              # width, height
        linewidth=2, edgecolor=edgecolor, facecolor='none'
        )
        ax.add_patch(rect)

        if plot_info == "coords":
            text_data = f"({spot_x},{spot_y})"
        elif plot_info == "indices":
            text_data = f"{i}"
        else:
            text_data = ""
        ax.text(
            spot_x, spot_y - box_size - 5, text_data,
            color='yellow', fontsize=8, ha='center', va='bottom', weight='bold'
        )

    plt.title("Detected Spots with Boxes")
    plt.show()

def draw_boundaries_around_spots_2d(image, array_size, plot_info="indices"):
    """
    Draws rectangular boundaries around detected spots in an image and annotates them.
    This function detects spots in the provided image, draws a box around each spot,
    and annotates each box with either the spot's coordinates or its row/column indices.
    The central spot (row 5, col 5) is highlighted in blue, while all others are red.
    Args:
        image (np.ndarray): The input image in which to detect and annotate spots.
        array_size (int): The size of the 2D array of spots (e.g., 11 for an 11x11 grid).
        plot_info (str, optional): Determines the annotation for each spot.
            - "indices": Annotates with row and column indices (default).
            - "coords": Annotates with x and y coordinates.
            - Any other value: No annotation.
    Returns:
        None. Displays the image with annotated spot boundaries using matplotlib.
    """

    spot_indices = find_spots(image, array_size**2)
    si_2d = get_2d_spot_indices(spot_indices)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='viridis')
    box_size = 15  # Half-width of the box (adjust as needed)

    for row in range(si_2d.shape[0]):
        for col in range(si_2d.shape[1]):
            spot_y, spot_x = si_2d[row, col]
            edgecolor = "blue" if row == array_size // 2 and col == array_size // 2 else "red"
            rect = patches.Rectangle(
            (spot_x - box_size, spot_y - box_size),  # (x, y) of lower left
            2 * box_size, 2 * box_size,              # width, height
            linewidth=2, edgecolor=edgecolor, facecolor='none'
            )
            ax.add_patch(rect)

            if plot_info == "coords":
                text_data = f"{int(spot_x)},{int(spot_y)}"
                fs = 4
            elif plot_info == "indices":
                text_data = f"{row},{col}"
                fs = 8
            else:
                text_data = ""
                fs = 8
            ax.text(
                spot_x, spot_y - box_size - 5, text_data,
                color='yellow', fontsize=fs, ha='center', va='bottom', weight='bold'
            )

    plt.title("Detected Spots with Boxes")
    plt.show()

def gaussian_plus_uniform(xy, A, x0, y0, sigma_x, sigma_y, C):
    x, y = xy
    g = A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) + C 
    return g.ravel()

def estimate_gaussian(img, intensity_guess=None, x_guess=None, y_guess=None, plot_image=False, print_output=False):
    """
    Estimate the parameters of a Gaussian function plus a uniform background from an image.

    Args:
        img (np.ndarray): The input image data.
        intensity_guess (float, optional): Initial guess for the peak intensity. Defaults to None.
        x_guess (int, optional): Initial guess for the x-coordinate of the peak. Defaults to None (center of the image).
        y_guess (int, optional): Initial guess for the y-coordinate of the peak. Defaults to None (center of the image).
        plot_image (bool, optional): Whether to plot the original and fitted images. Defaults to False.
    Returns:
        tuple: Fitted parameters (A, x0, y0, sigma_x, sigma_y, C). Note: x0 is left-right, y0 is up-down
        int: The beam waist in pixels in the x-direction
        int: The beam waist in pixels in the y-direction
    """
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    x, y = np.meshgrid(x, y)
    data = img.ravel()
    if x_guess is None:
        x_guess = img.shape[0] // 2
    if y_guess is None:
        y_guess = img.shape[1] // 2
    if intensity_guess is None:
        intensity_guess = np.max(img)
    initial_guess = (intensity_guess, x_guess, y_guess, 5, 5, 0)

    params, params_covariance = curve_fit(gaussian_plus_uniform, (x, y), data, p0=initial_guess)

    fitted_image = gaussian_plus_uniform((x, y), *params).reshape(img.shape[0], img.shape[1])
    if plot_image:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='viridis')
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(fitted_image, cmap='viridis')
        plt.title('Fitted Gaussian')
        plt.colorbar()
        
        plt.show()

    A, x0, y0, sigma_x, sigma_y, C = params
    beam_waist_x = 2 * sigma_x
    beam_waist_y = 2 * sigma_y
    
    if print_output:
        print(f"Fitted parameters:\nA: {A:.2f}\nx0: {x0:.2f}\ny0: {y0:.2f}\nsigma_x: {sigma_x:.2f}\nsigma_y: {sigma_y:.2f}\nC: {C:.2f}")
        # For a Gaussian beam, the "beam waist" (radius at 1/e^2 intensity) is 2*sigma
        pixel_size = 1.85e-6
        print(f"Beam waist (x): {beam_waist_x:.2f} pixels = {beam_waist_x * pixel_size*10e6:.2e} microns")
        print(f"Beam waist (y): {beam_waist_y:.2f} pixels = {beam_waist_y * pixel_size*10e6:.2e} microns")

    return params, beam_waist_x, beam_waist_y


def get_gaussian_estimate(img, x, y, r):
    """
    Given an image and coordinates (x, y), estimate the Gaussian parameters of a spot
    in the region image[x-r:x+r, y-r:y+r].

    Helper function for compute array uniformity
    Args:
        img (_type_): image to analyze
        x (_type_): center x coordinate
        y (_type_): center y coordinate
        r (_type_): half-length of square region

    Raises:
        ValueError: if gaussian fails to find a fit

    Returns:
        _type_: gaussian fit params, and beam waist bw_x and bw_y
    """
    spot_img = img[y-r:y+r, x-r:x+r]
    try:
        params, bw_x, bw_y = estimate_gaussian(spot_img, plot_image=False)
    except Exception as e:
        raise ValueError(f"Error estimating Gaussian for spot at ({x}, {y}): {e}")
    if params[0] < 10:
        raise ValueError("Error in Gaussian fit, intensity too low")

    return params, bw_x, bw_y
def compute_array_uniformity(img, num_peaks):
    """
    Compute the uniformity of an optical array, including intensities and beam waists

    Args:
        img (np.ndarray): The input image data.
        num_peaks (int): Number of peaks to find in the image.

    Returns:
        data: [gaussain intensities, total intensities], [beam_waist_x, beam_waist_y]
    """
    spot_indices = find_spots(img, num_peaks)
    si_2d = get_2d_spot_indices(spot_indices)

    # set parameters and initialize arrays for data to return
    border_size = 25
    gaussian_intensities = np.empty((si_2d.shape[0], si_2d.shape[1]))
    total_intensities = np.empty((si_2d.shape[0], si_2d.shape[1]))
    bw_x_data = np.empty((si_2d.shape[0], si_2d.shape[1]))
    bw_y_data = np.empty((si_2d.shape[0], si_2d.shape[1]))


    # first, perform the fit on the center spot of the array
    center_spot_y, center_spot_x = int(si_2d[5, 5, 0]), int(si_2d[5, 5, 1])
    params, bw_x, bw_y = get_gaussian_estimate(img, center_spot_x, center_spot_y, border_size)
    region_size_radius = math.ceil(max(bw_x, bw_y)*2.5)
    for row in range(si_2d.shape[0]):
        for col in range(si_2d.shape[1]):
            spot_y, spot_x = int(si_2d[row, col, 0]), int(si_2d[row, col, 1])
            params, bw_x, bw_y = get_gaussian_estimate(img, spot_x, spot_y, border_size)
            center_x, center_y = round(params[1]), round(params[2])
            center_x_original = spot_x-border_size+center_x
            center_y_original = spot_y-border_size+center_y
            spot_img_new = img[center_y_original-region_size_radius:center_y_original+region_size_radius, center_x_original-region_size_radius:center_x_original+region_size_radius]

            total_intensities[row, col] = np.sum(spot_img_new)
            gaussian_intensities[row, col] = params[0]
            bw_x_data[row, col] = bw_x
            bw_y_data[row, col] = bw_y

    return [gaussian_intensities, total_intensities], [bw_x_data, bw_y_data]

def hdr_merge(images, exposure_times, bit_depth=8):
    """Simple HDR merge: images is a list of 2-D arrays, exposure_times list of floats.

    Args:
        images (list of np.ndarray): List of images to merge.
        exposure_times (list of float): Corresponding exposure times for each image.
        bit_depth (int, optional): Number of bits per pixel in the images. For example, 8 for 8-bit images with ranges 0-255. Defaults to 8.

    """
    images_stacked = np.stack(images)
    images_stacked_filtered = np.where(images_stacked >= (2**bit_depth - 1), 0, images_stacked)

    t = exposure_times[:, np.newaxis, np.newaxis]  # reshape for broadcasting
    # Normalize images by exposure times
    normalized_images = images_stacked_filtered / t

    # for each pixel in the image, take the first nonzero value from the largest exposure time possible
    final_image = np.zeros_like(images[0], dtype=float)
    for image in normalized_images[::-1]:
        
        mask = (final_image == 0) & (image > 0)
        final_image[mask] = image[mask]
    return final_image

def compute_nearest_neighbor_crosstalk(y:int, x:int, spot_array_size, full_array_img:np.ndarray, images: list[np.ndarray], exposure_times: np.ndarray[float], spot_bucket_hw: int, r:int) -> np.ndarray:
    """
    Computes the nearest neighbor crosstalk for a given spot (x, y) in a list of images.

    Args:
        y (int): y-coordinate of the spot in the peaks indices array (vertical, first indice in numpy array).
        x (int): x-coordinate of the spot in the peaks indices array (horizontal, second indice in numpy array).
        spot_array_size (int): Size of the 2D array of spots (e.g., 11 for an 11x11 grid).
        full_array_img (np.ndarray): The full array image, with all spots clearly visible. Must be same shape as images.
        images (list[np.ndarray]): List of images to analyze.
        spot_bucket_hw (int): Half-width of the spot bucket.
        r (int): Radius around the pixel to consider for crosstalk.

    Returns:
        np.ndarray: Crosstalk values for each image.
    """
    final_image = hdr_merge(images, exposure_times)
    print("Final Image assembled")
    peaks_indices = get_2d_spot_indices(find_spots(full_array_img, spot_array_size**2), array_size=spot_array_size)
    
    middle_y, middle_x = peaks_indices[y, x]
    x1, x2 = int(middle_x - spot_bucket_hw), int(middle_x + spot_bucket_hw + 1)
    y1, y2 = int(middle_y - spot_bucket_hw), int(middle_y + spot_bucket_hw + 1)

    # draw the region of middle bin intensity on top of the hdr image
    # Draw the region of middle bin intensity on top of the HDR image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(final_image, cmap='viridis')
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor='cyan',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title("Middle Bin Region on HDR Image")
    plt.show()
    middle_bin_intensity = np.max(final_image[y1:y2, x1:x2])
    crosstalk_I_ratio = np.zeros((2*r+1, 2*r+1), dtype=np.float64)
    crosstalk_db = np.zeros((2*r+1, 2*r+1), dtype=np.float64)
    print("middle bin intensity", middle_bin_intensity)
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if i == 0 and j == 0:
                continue
            if 0 <= y+i < peaks_indices.shape[0] and 0 <= x+j < peaks_indices.shape[1]:
                spot_location_y, spot_location_x = peaks_indices[y+i, x+j]
                x1, x2 = int(spot_location_x - spot_bucket_hw), int(spot_location_x + spot_bucket_hw + 1)
                y1, y2 = int(spot_location_y - spot_bucket_hw), int(spot_location_y + spot_bucket_hw + 1)
                if x1 < 0 or x2 > final_image.shape[1] or y1 < 0 or y2 > final_image.shape[0]:
                    raise ValueError(f"Spot at ({spot_location_x}, {spot_location_y}) is out of bounds for the final image shape {final_image.shape}. \
                        \n Coords are {x1, x2}, {y1, y2} and final image shape is {final_image.shape}")
                bin_intensity = np.max(final_image[y1:y2, x1:x2])
                crosstalk_I_ratio[i+r, j+r] = bin_intensity / middle_bin_intensity
                crosstalk_db[i+r, j+r] = np.log10(bin_intensity / middle_bin_intensity) * 10
    return crosstalk_I_ratio, crosstalk_db

def single_spot_crosstalk(y:int, x:int, peaks_indices, hdr_img: np.ndarray[float], spot_bucket_hw: int, r:int) -> np.ndarray:
    """
    Computes the nearest neighbor crosstalk for a given spot (x, y) in a list of images.

    Args:
        y (int): y-coordinate of the spot in the peaks indices array (vertical, first indice in numpy array).
        x (int): x-coordinate of the spot in the peaks indices array (horizontal, second indice in numpy array).
        spot_array_size (int): Size of the 2D array of spots (e.g., 11 for an 11x11 grid).
        full_array_img (np.ndarray): The full array image, with all spots clearly visible. Must be same shape as images.
        images (list[np.ndarray]): List of images to analyze.
        spot_bucket_hw (int): Half-width of the spot bucket.
        r (int): Radius around the pixel to consider for crosstalk.

    Returns:
        np.ndarray: Crosstalk values for each image.
    """
    
    middle_y, middle_x = peaks_indices[y, x]
    x1, x2 = int(middle_x - spot_bucket_hw), int(middle_x + spot_bucket_hw + 1)
    y1, y2 = int(middle_y - spot_bucket_hw), int(middle_y + spot_bucket_hw + 1)

    middle_bin_intensity = np.max(hdr_img[y1:y2, x1:x2])
    crosstalk_I_ratio = np.zeros((2*r+1, 2*r+1), dtype=np.float64)
    crosstalk_db = np.zeros((2*r+1, 2*r+1), dtype=np.float64)
    counts_array = np.zeros((2*r+1, 2*r+1), dtype=np.int64)
    print("middle bin intesnity is ", np.max(middle_bin_intensity))
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if i == 0 and j == 0:
                continue
            if 0 <= y+i < peaks_indices.shape[0] and 0 <= x+j < peaks_indices.shape[1]:
                spot_location_y, spot_location_x = peaks_indices[y+i, x+j]
                x1, x2 = int(spot_location_x - spot_bucket_hw), int(spot_location_x + spot_bucket_hw + 1)
                y1, y2 = int(spot_location_y - spot_bucket_hw), int(spot_location_y + spot_bucket_hw + 1)
                if x1 < 0 or x2 > hdr_img.shape[1] or y1 < 0 or y2 > hdr_img.shape[0]:
                    raise ValueError(f"Spot at ({spot_location_x}, {spot_location_y}) is out of bounds for the final image shape {hdr_img.shape}. \
                        \n Coords are {x1, x2}, {y1, y2} and final image shape is {hdr_img.shape}")
                bin_intensity = np.max(hdr_img[y1:y2, x1:x2])
                crosstalk_I_ratio[i+r, j+r] = bin_intensity / middle_bin_intensity
                crosstalk_db[i+r, j+r] = np.log10(bin_intensity / middle_bin_intensity) * 10
                counts_array[i+r, j+r] += 1
    return crosstalk_I_ratio, crosstalk_db, counts_array

def average_nn_crosstalk_whole_array(
        spot_array_size: int,
        cropping_function: callable,
        full_array_img: np.ndarray,
        exposure_times: np.ndarray,
        dark_images: list[np.ndarray],
        spot_bucket_hw: int,
        r: int
    ):
    """
    Computes the average nearest-neighbor (NN) crosstalk for each spot in a 2D spot array, as well as the full crosstalk matrices.
    This function processes a set of images corresponding to each spot in a spot array, subtracts dark images, merges exposures using HDR, and calculates crosstalk ratios and decibel values for each spot and its neighbors. It then averages the crosstalk values for the four nearest neighbors of each spot.
    Args:
        spot_array_size (int): The size (width/height) of the square spot array.
        cropping_function (callable): Function to crop the images before merging so that it has the same shape as full_array_img.
        full_array_img (np.ndarray): The full image containing all spots.
        exposure_times (np.ndarray): Array of exposure times corresponding to the images.
        dark_images (list[np.ndarray]): List of dark images to subtract from each exposure.
        spot_bucket_hw (int): Half-width of the spot region to analyze around each peak.
        r (int): Radius (in spot units) to consider for crosstalk calculation.
    Returns:
        tuple:
            crosstalk_I_ratios (np.ndarray): 2D array of average crosstalk intensity ratios for each neighbor offset.
            crosstalk_dbs (np.ndarray): 2D array of average crosstalk values in decibels for each neighbor offset.
            avg_nn_crosstalk (np.ndarray): 2D array of average nearest-neighbor crosstalk (in dB) for each spot in the array.
    """
    
    peaks_indices = get_2d_spot_indices(find_spots(full_array_img, spot_array_size**2), array_size=spot_array_size)
    
    # Initialize crosstalk matrices
    crosstalk_I_ratios = np.zeros((2*r+1, 2*r+1), dtype=np.float64)
    crosstalk_dbs = np.zeros((2*r+1, 2*r+1), dtype=np.float64)
    counter = np.zeros((2*r+1, 2*r+1), dtype=np.int64)
    avg_nn_crosstalk = np.zeros((spot_array_size, spot_array_size), dtype=np.float64)
    
    for row in range(peaks_indices.shape[0]):
        for col in range(peaks_indices.shape[1]):
            print(f"Working on spot at (row, col): {row}, {col}")
            images_dir = f"./output/small_array/individual_spots/trial_1/spot_{row}_{col}"
            lights_imgs = []
            t_unsorted = np.zeros_like(exposure_times)
            for i, filename in enumerate(os.listdir(images_dir)):
                if filename.endswith(".npy"):
                    exposure_time = int(filename.split("_")[1].split(".")[0])
                    image = np.load(os.path.join(images_dir, filename))
                    
                    lights_imgs.append(image)
                    t_unsorted[i] = exposure_time
                    
                    if exposure_time not in exposure_times:
                        raise ValueError(f"Exposure time {exposure_time} not found in exposure_times array.")
                else:
                    continue
            # take images
            indices = np.argsort(t_unsorted)
            lights_imgs = [np.array(lights_imgs[i]) for i in indices]
            imgs_minus_dark = [img - dark for img, dark in zip(lights_imgs, dark_images)]
            imgs = [np.clip(img, 0, None) for img in imgs_minus_dark]

            imgs_cropped = [cropping_function(img) for img in imgs]
            final_img = hdr_merge(imgs_cropped, exposure_times)
            
            c_I, c_DB, counts = single_spot_crosstalk(y=row, x=col, peaks_indices=peaks_indices, hdr_img=final_img, spot_bucket_hw=spot_bucket_hw, r=r)
            crosstalk_I_ratios += c_I
            crosstalk_dbs += c_DB
            counter += counts

            # find the four nearest spots to the center
            val1 = c_DB[r-1, r] # up
            val2 = c_DB[r+1, r] # down
            val3 = c_DB[r, r-1] # left
            val4 = c_DB[r, r+1] # up

            # average them, while taking into account values that are zero
            avg_nn_crosstalk[row, col] = (val1 + val2 + val3 + val4) / np.count_nonzero([val1, val2, val3, val4])
            print("avg nn crosstalk", avg_nn_crosstalk[row, col])
    # Average the crosstalk values, handling division by zero because the middle pixel is always zero
    with np.errstate(divide='ignore', invalid='ignore'):
        crosstalk_I_ratios = np.where(counter == 0, 0, crosstalk_I_ratios / counter)
        crosstalk_dbs = np.where(counter == 0, 0, crosstalk_dbs / counter)
    return crosstalk_I_ratios, crosstalk_dbs, avg_nn_crosstalk


def get_band_profile(band_image, plot_images, half_width=10, logscale=True):
    row_max = np.argmax(band_image.sum(axis=1))
    final_band = band_image[row_max-half_width : row_max+half_width+1, :]
    horizontal_lines = [row_max - half_width, row_max + half_width]  # for plotting horizontal lines

    # ---------- 3. collapse to 1-D profile ----------
    profile = final_band.max(axis=0)      # shape (columns,) # TODO: Decide on sum, mean, or cross section (paper uses cross section)

    # ---------- 4. normalize ----------
    profile = profile / profile.max()

    # optional: smooth just for presentation (savgol_filter keeps peaks sharp)
    profile_smoothed = savgol_filter(profile, window_length=11, polyorder=2)

    # ---------- 5. plot ----------
    fig, axs = plt.subplots(len(plot_images)+1, 1, figsize=(10, 10))
    x_pixels = np.arange(profile.size)
    for i in range(len(plot_images)):
        axs[i].imshow(plot_images[i], cmap='Blues')
        for y in horizontal_lines:
            axs[i].axhline(y, color='red', linestyle='--', lw=0.5)
    # cosmetic touches like in Fig. 3a
    axs[-1].plot(x_pixels, profile_smoothed, lw=1.2, color="tab:blue")
    axs[-1].set_xlabel("camera column (pixels)")
    axs[-1].set_ylabel("normalized intensity")
    if logscale:
        axs[-1].set_yscale("log")             # comment this out for linear axis
    axs[-1].grid(True, which="both", ls="--", lw=0.4)
    
    axs[-1].set_xlim(axs[0].get_xlim())
    plt.tight_layout()
    plt.show()
        