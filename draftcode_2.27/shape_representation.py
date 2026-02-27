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





mask = segment_fruit(img)
mask_refined = refine_mask(mask)

shape_feats = compute_shape_features(mask_refined)
shape_feats
