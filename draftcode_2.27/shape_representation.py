import cv2
import numpy as np
from skimage import morphology
from pyefd import elliptic_fourier_descriptors
import mahotas
import matplotlib.pyplot as plt



def load_and_normalize(path, target_size=(90, 90)):
    # Read image from file
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    
    # Convert BGR  to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to a fixed resolution
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    #  illumination normalization
    img_float = img_resized.astype(np.float32)
    img_norm = cv2.normalize(img_float, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)
    return img_norm




def segment_fruit(img_rgb):
    """
    Segment the fruit from the background using HSV thresholding.
    Input: normalized RGB image (90x90)
    Output: binary mask (0 = background, 255 = fruit)
    """
    # Convert RGB → HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Example thresholds (tune for your dataset)
    lower = np.array([10, 40, 40])     # lower HSV bound
    upper = np.array([35, 255, 255])   # upper HSV bound

    # Create mask: fruit = 255, background = 0
    mask = cv2.inRange(hsv, lower, upper)

    return mask





# Segmentation converts the normalized RGB image into a binary mask using HSV thresholding. 
# This mask identifies which pixels belong to the fruit. It must be done before shape analysis because
# shape features (contour, area, perimeter, EFDs) depend entirely on the segmented object.

def refine_mask(mask):
    """
    Clean the binary mask by removing noise and filling holes.
    Input: raw mask from segmentation (0 or 255)
    Output: refined mask (0 or 255)
    """
    # Convert to boolean for skimage operations
    mask_bool = mask.astype(bool)

    # Fill small holes inside the fruit
    mask_filled = morphology.remove_small_holes(mask_bool, area_threshold=200)

    # Remove small noisy objects
    mask_clean = morphology.remove_small_objects(mask_filled, min_size=200)

    # Convert back to uint8 (0 or 255)
    mask_clean = (mask_clean.astype(np.uint8)) * 255

    return mask_clean



# Mask refinement removes noise and fills holes in the segmented fruit region. 
# This ensures the mask is smooth, connected, and stable before extracting shape features. 
# Clean masks produce accurate contours, which are essential for EFDs, curvature, and geometric measurements.


def compute_shape_features(mask_clean):
    """
    Extract advanced shape features from refined mask.
    """
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        raise ValueError("No contour found in mask.")

    cnt = max(contours, key=cv2.contourArea)

    # -------------------------
    # Boundary smoothing
    # -------------------------
    pts = cnt[:, 0, :].astype(np.float32)

    # Gaussian smoothing along contour
    ksize = 5
    pts_smooth = np.copy(pts)
    pts_smooth[:, 0] = cv2.GaussianBlur(pts[:, 0], (ksize, 1), 0).flatten()
    pts_smooth[:, 1] = cv2.GaussianBlur(pts[:, 1], (ksize, 1), 0).flatten()

    cnt_smooth = pts_smooth.reshape(-1, 1, 2).astype(np.int32)

    # -------------------------
    # Geometric features
    # -------------------------
    area = cv2.contourArea(cnt_smooth)
    perimeter = cv2.arcLength(cnt_smooth, True)
    perimeter_norm = perimeter / (np.sqrt(area) + 1e-6)
    circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

    # -------------------------
    # Solidity
    # -------------------------
    hull = cv2.convexHull(cnt_smooth)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    # -------------------------
    # Eccentricity
    # -------------------------
    if len(cnt_smooth) >= 5:
        ellipse = cv2.fitEllipse(cnt_smooth)
        (_, axes, _) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis / (major_axis + 1e-6))**2)
    else:
        eccentricity = 0.0

    # -------------------------
    # Curvature (smoothed)
    # -------------------------
    dx = np.gradient(pts_smooth[:, 0])
    dy = np.gradient(pts_smooth[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.mean(
        np.abs(dx * ddy - dy * ddx) /
        (dx * dx + dy * dy + 1e-6) ** 1.5
    )

    # -------------------------
    # Elliptic Fourier Descriptors
    # -------------------------
    efd = elliptic_fourier_descriptors(pts_smooth, order=10, normalize=True)
    efd_flat = efd.flatten()

    # -------------------------
    # Radial Distance Signature
    # -------------------------
    moments = cv2.moments(cnt_smooth)
    cx = moments["m10"] / (moments["m00"] + 1e-6)
    cy = moments["m01"] / (moments["m00"] + 1e-6)

    radial = np.sqrt((pts_smooth[:, 0] - cx)**2 +
                     (pts_smooth[:, 1] - cy)**2)

    radial_norm = radial / (np.mean(radial) + 1e-6)

    # FFT of radial signature (keep low freq)
    radial_fft = np.abs(np.fft.fft(radial_norm))
    radial_descriptor = radial_fft[:15]  # low-frequency components

    # -------------------------
    # Zernike Moments
    # -------------------------
    radius = min(mask_clean.shape) // 2
    zernike = mahotas.features.zernike_moments(mask_clean, radius, degree=8)

    # -------------------------
    # Return upgraded feature set
    # -------------------------
    shape_features = {
        "area": area,
        "perimeter": perimeter,
        "perimeter_norm": perimeter_norm,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "curvature": curvature,
        "efd": efd_flat,
        "radial_signature": radial_descriptor,
        "zernike": zernike,
    }

    return shape_features









def circular_mean_std(h):
    """
    This helps convert the Hue measurements to radians due to the scale of 0-180 having 0 and 180 both be red,
    this way an apple with values of 2 and 179 wont average to 90(green hue) and will instead average to a red hue in radians
    """
    #convert to radians
    theta = (h.astype(np.float32) / 180.0) * 2.0 * np.pi

    #convert theta angles to unit circle locations
    sin_vals = np.sin(theta)
    cos_vals = np.cos(theta)

    sin_m = np.mean(sin_vals)
    cos_m = np.mean(cos_vals)

    #give averaged angle vector
    mean_theta = np.arctan2(sin_m, cos_m)

    #catch negatives
    if mean_theta < 0:
        mean_theta += 2.0 * np.pi

    #test closeness of values
    R = np.sqrt(sin_m**2 + cos_m**2)
    R = np.clip(R, 1e-6, 1.0)

    #get std
    circ_std = np.sqrt(-2.0 * np.log(R))

    #convert back to hue scale for final results
    # for feature space, sin/cos are better than mean_h because they preserve wrap-around (red near 0 and 180)
    std_h  = (circ_std / (2.0 * np.pi)) * 180.0

    return sin_m, cos_m, std_h


def compute_colour_features(img_rgb, mask_clean):
    """
    Ripeness features using circular Hue stats:
    [H_sin_mean, H_cos_mean, S_mean, V_mean, H_circ_std, S_std, V_std]
    """
    #get mask
    mask_bool = mask_clean.astype(bool)

    #grab h(hue) s(saturation) v(value) values to better deal with brighness of colors.
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    fruit_pixels = hsv[mask_bool]

    if fruit_pixels.size == 0:
        raise ValueError("No fruit pixels found for HSV colour features.")

    H = fruit_pixels[:, 0]
    S = fruit_pixels[:, 1].astype(np.float32)
    V = fruit_pixels[:, 2].astype(np.float32)

    # Hue is circular, so keep it on the unit circle (sin/cos) instead of raw mean hue
    h_sin_mean, h_cos_mean, h_std = circular_mean_std(H)

    s_mean, v_mean = float(S.mean()), float(V.mean())
    s_std,  v_std  = float(S.std()),  float(V.std())

    return np.array([h_sin_mean, h_cos_mean, s_mean, v_mean, h_std, s_std, v_std], dtype=np.float32)




# Step 5 extracts ripeness‑related colour features from the fruit region. Using the refined mask, 
# we isolate only the fruit pixels and compute the mean and standard deviation of the HSV channels. 
# The mean values capture overall colour (ripeness), while the standard deviations capture texture and colour variation 
# (spots, bruising, over‑ripeness). These features complement the shape descriptors and are essential for classification.



def build_feature_vector(shape_features, colour_features):
    """
    Combine shape and colour features into a single classification-ready vector.
    """
    
    # Scalars only
    geom = np.array([
        shape_features["area"],
        shape_features["perimeter"],
        shape_features["perimeter_norm"],
        shape_features["circularity"],
        shape_features["solidity"],
        shape_features["eccentricity"],
        shape_features["curvature"]
    ], dtype=np.float32)

    # Vector features
    efd = np.array(shape_features["efd"], dtype=np.float32).flatten()
    radial = np.array(shape_features["radial_signature"], dtype=np.float32).flatten()
    zernike = np.array(shape_features["zernike"], dtype=np.float32).flatten()

    colour = np.array(colour_features, dtype=np.float32).flatten()

    # Combine everything
    feature_vector = np.hstack([
        geom,
        efd,
        radial,
        zernike,
        colour
    ])

    return feature_vector




def visualize_mask(img_rgb, mask, mask_refined=None):
    
    ncols = 3 if mask_refined is None else 4
    
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, ncols, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    # Raw mask
    plt.subplot(1, ncols, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Raw Mask")
    plt.axis("off")
    
    # Refined mask (if available)
    if mask_refined is not None:
        plt.subplot(1, ncols, 3)
        plt.imshow(mask_refined, cmap="gray")
        plt.title("Refined Mask")
        plt.axis("off")
        
        # Overlay
        overlay = img_rgb.copy()
        overlay[mask_refined == 0] = overlay[mask_refined == 0] * 0.3
        
        plt.subplot(1, ncols, 4)
        plt.imshow(overlay)
        plt.title("Mask Overlay")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# test run
img = load_and_normalize("apple.jpg")


mask = segment_fruit(img)


mask_refined = refine_mask(mask)



shape_feats = compute_shape_features(mask_refined)

colour_feats = compute_colour_features(img, mask_refined)

covariates = build_feature_vector(shape_feats, colour_feats)
print(covariates)

visualize_mask(img, mask, mask_refined)
