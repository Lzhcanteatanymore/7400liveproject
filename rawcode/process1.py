import cv2
import numpy as np
from skimage import morphology
from pyefd import elliptic_fourier_descriptors

conda install -c conda-forge pyefd

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

img = load_and_normalize("/Users/shouhardyo/Desktop/uiowa/7400/live/test.jpeg")
print(img.shape)


# STEP 2- SEGMENTATION

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

img = load_and_normalize("/Users/shouhardyo/Desktop/uiowa/7400/live/test.jpeg")
mask = segment_fruit(img)
print(mask.shape)

# Segmentation converts the normalized RGB image into a binary mask using HSV thresholding. 
# This mask identifies which pixels belong to the fruit. It must be done before shape analysis because
# shape features (contour, area, perimeter, EFDs) depend entirely on the segmented object.


# STEP 3: MASK REFINEMENT


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
img = load_and_normalize("/Users/shouhardyo/Desktop/uiowa/7400/live/test.jpeg")
mask = segment_fruit(img)
mask_refined = refine_mask(mask)
print(mask_refined.shape)

# Mask refinement removes noise and fills holes in the segmented fruit region. 
# This ensures the mask is smooth, connected, and stable before extracting shape features. 
# Clean masks produce accurate contours, which are essential for EFDs, curvature, and geometric measurements.


# STEP 4: Shape representation
def compute_shape_features(mask_clean):
    """
    Extract shape features from the refined binary mask.
    Features: area, perimeter, curvature, EFDs.
    """
    # Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        raise ValueError("No contour found in mask.")
    
    # Use the largest contour (the fruit)
    cnt = max(contours, key=cv2.contourArea)
    
    # -------------------------
    # 1. Geometric features
    # -------------------------
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # -------------------------
    # 2. Curvature (approximate)
    # -------------------------
    pts = cnt[:, 0, :]  # shape (N, 2)
    
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.mean(
        np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy + 1e-6) ** 1.5
    )
    
    # -------------------------
    # 3. Elliptic Fourier Descriptors (EFDs)
    # -------------------------
    efd = elliptic_fourier_descriptors(pts, order=10, normalize=True)
    efd_flat = efd.flatten()
    
    # -------------------------
    # Return all features
    # -------------------------
    shape_features = {
        "area": area,
        "perimeter": perimeter,
        "curvature": curvature,
        "efd": efd_flat,
    }
    
    return shape_features
img = load_and_normalize("/Users/shouhardyo/Desktop/uiowa/7400/live/test.jpeg")
mask = segment_fruit(img)
mask_refined = refine_mask(mask)

shape_feats = compute_shape_features(mask_refined)
print("Area:", shape_feats["area"])
print("Perimeter:", shape_feats["perimeter"])
print("Curvature:", shape_feats["curvature"])
print("EFD length:", len(shape_feats["efd"]))

# Step 4 extracts shape features from the refined mask. We compute geometric features 
# (area, perimeter), curvature (boundary smoothness), and Elliptic Fourier Descriptors (EFDs), 
# which provide a compact and rotation‑invariant representation of the fruit’s contour. 
# These features are essential for classification and depend entirely on the quality of the segmentation and mask refinement steps.


# STEP 5: RIPENESS
def compute_colour_features(img_rgb, mask_clean):
    """
    Compute ripeness-related colour features from the fruit region.
    Features: mean and std of R, G, B channels.
    """
    # Convert mask to boolean for indexing
    mask_bool = mask_clean.astype(bool)

    # Extract only fruit pixels
    fruit_pixels = img_rgb[mask_bool]

    if fruit_pixels.size == 0:
        raise ValueError("No fruit pixels found for colour features.")

    # Mean colour (R, G, B)
    mean_rgb = fruit_pixels.mean(axis=0)

    # Standard deviation (texture / ripeness variation)
    std_rgb = fruit_pixels.std(axis=0)

    # Final feature vector: [R_mean, G_mean, B_mean, R_std, G_std, B_std]
    colour_features = np.concatenate([mean_rgb, std_rgb])

    return colour_features
img = load_and_normalize("/Users/shouhardyo/Desktop/uiowa/7400/live/test.jpeg")
mask = segment_fruit(img)
mask_refined = refine_mask(mask)
shape_feats = compute_shape_features(mask_refined)

colour_feats = compute_colour_features(img, mask_refined)
print("Colour features:", colour_feats)
print("Length:", len(colour_feats))
# Step 5 extracts ripeness‑related colour features from the fruit region. Using the refined mask, 
# we isolate only the fruit pixels and compute the mean and standard deviation of the R, G, and B channels. 
# The mean values capture overall colour (ripeness), while the standard deviations capture texture and colour variation 
# (spots, bruising, over‑ripeness). These features complement the shape descriptors and are essential for classification.
Colour features: [180.10165403 107.25430737  42.12474156  34.77950142  32.47676538
  33.30600165]
Length: 6
# STEP 6 : Feature extraction
def build_feature_vector(shape_features, colour_features):
    """
    Combine shape and colour features into a single classification-ready vector.
    """
    # Geometric features
    geom = np.array([
        shape_features["area"],
        shape_features["perimeter"],
        shape_features["curvature"],
    ])

    # EFDs (already flattened)
    efd = shape_features["efd"]

    # Final feature vector
    feature_vector = np.hstack([geom, efd, colour_features])

    return feature_vector
img = load_and_normalize("/Users/shouhardyo/Desktop/uiowa/7400/live/test.jpeg")
mask = segment_fruit(img)
mask_refined = refine_mask(mask)
shape_feats = compute_shape_features(mask_refined)
colour_feats = compute_colour_features(img, mask_refined)

feature_vector = build_feature_vector(shape_feats, colour_feats)
print("Feature vector length:", len(feature_vector))
print("Feature vector:", feature_vector)



# Step 6 combines all extracted features — geometric features (area, perimeter, curvature), 
# shape descriptors (EFDs), and colour statistics (mean and standard deviation of RGB channels) — into a single feature vector. 
# This vector fully represents the fruit’s shape and ripeness characteristics and is ready to be used by any machine learning classifier.

# OUTPUT
# First 3 values → geometric features
# 2797.0      # area
# 246.107646  # perimeter
# 0.259086741 # curvature

# These describe the size, boundary length, and smoothness of the fruit.


# Next 40 values → EFD coefficients
# These 40 numbers encode the entire contour shape of the fruit in a rotation‑invariant way.
# They are the most powerful shape descriptors in our pipeline.


# Last 6 values → colour/ripeness features

# 180.101654   # R_mean
# 107.254307   # G_mean
# 42.1247416   # B_mean
# 34.7795014   # R_std
# 32.4767654   # G_std
# 33.3060017   # B_std

# These capture:
# overall colour (ripeness)
# colour variation (spots, bruising, over‑ripeness)





conda install -c conda-forge opencv
