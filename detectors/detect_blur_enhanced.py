from imutils import paths
import argparse
import cv2
import os
import numpy as np

# This script contains more advanced ways of detecting blurriness.
# Steps: 
# 1. Open up the terminal
# 2. Choose from one of the modes: 'center', 'patch_max', 'patch_percentile'
# 2. Type this line into the terminal window, and hit "enter": 
# "python detect_blur_enhanced.py --images path-to-the-image-folder --mode 'chosen-mode'"
# (Add other arguments as needed to adjust.)
# Recommended parameters:
# Use --mode 'center' only when major objects are close to the center
# --mode 'patch_max' --threshold 450 --clear_threshold 670 
# --mode 'patch_percentile' --percentile 80.0 --threshold 361 --clear_threshold 515 

def variance_of_laplacian(image):
    L = cv2.Laplacian(image, cv2.CV_64F)
    return L.var()

def gaussian_weight_mask(h, w, sigma=None):
    if sigma is None:
        sigma = 0.5 * min(h, w)   # default sigma (pixels)
    # coords centered
    xv = np.arange(w) - (w - 1) / 2.0
    yv = np.arange(h) - (h - 1) / 2.0
    xv, yv = np.meshgrid(xv, yv)
    g = np.exp(-(xv**2 + yv**2) / (2 * (sigma**2)))
    return g.astype(np.float64)

def weighted_variance_of_laplacian(image, weight_mask):
    # compute Laplacian once
    L = cv2.Laplacian(image, cv2.CV_64F).astype(np.float64)
    w = weight_mask.astype(np.float64)
    s = w.sum()
    if s == 0:
        return L.var()
    mu = (w * L).sum() / s
    var = (w * (L - mu)**2).sum() / s
    return float(var)

def patch_variances_of_laplacian(image, grid):
    # compute Laplacian once
    L = cv2.Laplacian(image, cv2.CV_64F).astype(np.float64)
    h, w = image.shape[:2]
    vars_list = []
    for r in range(grid):
        y0 = int(r * h / grid)
        y1 = int((r + 1) * h / grid) if r < grid - 1 else h
        for c in range(grid):
            x0 = int(c * w / grid)
            x1 = int((c + 1) * w / grid) if c < grid - 1 else w
            patch = L[y0:y1, x0:x1]
            # guard against empty
            if patch.size == 0:
                pv = 0.0
            else:
                pv = float(patch.var())
            vars_list.append(pv)
    return vars_list

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=350.0,
                help="blur threshold (below => blurry)")
ap.add_argument("-c", "--clear_threshold", type=float, default=500.0,
                help="clear threshold (above => clear)")
ap.add_argument("-m", "--mode", choices=['center', 'patch_max', 'patch_percentile'],
                default='patch_max', help="sharpness mode")
ap.add_argument("-g", "--grid", type=int, default=3,
                help="grid size (NxN) for patch methods")
ap.add_argument("-p", "--percentile", type=float, default=90.0,
                help="percentile for patch_percentile mode (e.g. 90)")
ap.add_argument("--sigma", type=float, default=None,
                help="sigma in pixels for center gaussian weight (None -> 0.5*min(h,w))")
args = vars(ap.parse_args())

blurry_out = "blurry.txt"
other_out = "other.txt"
clear_out = "clear.txt"

# open files
with open(blurry_out, "w") as bf, open(other_out, "w") as nf, open(clear_out, "w") as cf:
    for imagePath in paths.list_images(args["images"]):
        filename = os.path.basename(imagePath)

        image = cv2.imread(imagePath)
        if image is None:
            print(f"Warning: couldn't read {imagePath}, skipping.")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mode = args["mode"]
        if mode == 'center':
            h, w = gray.shape[:2]
            mask = gaussian_weight_mask(h, w, sigma=args["sigma"])
            fm = weighted_variance_of_laplacian(gray, mask)
            debug_val = fm
        elif mode == 'patch_max':
            vars_list = patch_variances_of_laplacian(gray, args["grid"])
            fm = max(vars_list) if vars_list else 0.0
            debug_val = fm
        elif mode == 'patch_percentile':
            vars_list = patch_variances_of_laplacian(gray, args["grid"])
            fm = float(np.percentile(vars_list, args["percentile"])) if vars_list else 0.0
            debug_val = fm
        else:
            # fallback to global
            fm = variance_of_laplacian(gray)
            debug_val = fm

        # classification using thresholds
        if fm < args["threshold"]:
            label = "Blurry"
            # bf.write(filename + "\n")
        elif fm > args["clear_threshold"]:
            label = "Clear"
            # cf.write(filename + "\n")
        else:
            label = "Other"
            # nf.write(filename + "\n")

        # Show the image - can be commented out. 
        # If you want to change to a better threshold, uncomment it and test on a smaller image set. 
        cv2.putText(image, "{} : {:.2f}".format(label, debug_val), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)  # press any key to continue

print("Done. Results written to:", blurry_out, other_out, clear_out)
