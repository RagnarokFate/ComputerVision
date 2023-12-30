# This is a sample Python script.
import PIL
import os
from PIL import Image
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import re
from tqdm.notebook import tqdm

# Find an existing implementation of SIFT (an implementation of detection and description and nothing more) and run it on each
# from the pictures in the puzzle. As you remember, the detector result, for each point of interest (IP), is a vector [t, r, y, x], where x and y are
# Position coordinates r and t are the scale and orientation, and the descriptor result is a 128-dimensional vector.
def SIFT_transform(image):
    sift = cv2.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(image, None)

    return keyPoints, descriptors


# Extract the matches that passed the test via ratio test
# using the matrix distances
def find_matrix_then_matches(des1, des2):
    from scipy.spatial.distance import cdist

    ratio_threshold = 0.8

    # Compute the Euclidean distances between the descriptors
    distances = cdist(des1, des2)

    # Find the indices and distances of the nearest and second nearest neighbors
    nn_indices, nn2_indices = zip(*[(np.argmin(d), np.argsort(d)[1]) for d in distances])
    nn_distances = distances[np.arange(len(nn_indices)), nn_indices]
    nn2_distances = distances[np.arange(len(nn2_indices)), nn2_indices]

    # Apply ratio test to filter out ambiguous matches
    ratios = nn_distances / nn2_distances
    matches = [cv2.DMatch(i, nn_indices[i], nn_distances[i]) for i, m in enumerate(ratios) if m < ratio_threshold]

    # Check if there are enough matches
    flag = len(matches) > 10

    return matches, flag

def CROP_IMAGE(IMG,Textfile_height,Textfile_width):
    # Convert the image to grayscale
    gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)

    # Find the contours of the image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_image = IMG[y:y + h, x:x + w]

    # Resize the image to the specified height and width
    resized_image = cv2.resize(cropped_image, (int(Textfile_height),int(Textfile_width)))

    # Return the resized image
    return resized_image

def apply_transformation(IMG, transformation_matrix):

    height, width = IMG.shape[:2]

    # Apply the transformation matrix to the image
    transformed_image = cv2.warpPerspective(IMG, transformation_matrix, (width, height))

    # Return the transformed image
    return transformed_image



# Find existing implementations of a minimal solver - one that uses 3 fits to fit an affine transformation
# And one that uses 4 matches for homography matching. This refers only to the solver whose input is only
# The coordinates of 3) or 4 (the pairs of points) (not a function that receives images, or all the matches).
def solver(source_pts, dist_pts, type_):
    if type_ == "affine":
        return cv2.getAffineTransform(source_pts, dist_pts)
    else:
        return cv2.findHomography(source_pts, dist_pts)


# Apply a simple RANSAC loop that using the solver and the calculation of the residuals. Perfect the RANSAC
# as you wish.

# def RANSAC_Implementation(matches, kp1, kp2, trans_type):
#     ransac_iterations = 2000
#     ransac_threshold = 5
#     src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#     # RANSAC loop
#     best_affine = None
#     best_inliers = None
#     if trans_type == "affine":
#         for i in range(ransac_iterations):
#             # Choose random 3 point correspondences
#             indices = np.random.choice(len(matches), 3, replace=False)
#             src_subset = src_points[indices]
#             dst_subset = dst_points[indices]
#
#             # Compute affine transformation from the 3 point correspondences
#             affine_transform = solver(src_subset, dst_subset, "affine")
#
#             # Calculate residuals for all correspondences
#             transformed_points = cv2.transform(src_points, affine_transform)
#             residuals = np.linalg.norm(dst_points - transformed_points, axis=-1)
#
#             # Find inliers within threshold
#             inliers = residuals < ransac_threshold
#
#             # Update best affine and inliers if current model is better
#             if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
#                 best_affine = affine_transform
#                 best_inliers = inliers
#     else:
#         for i in range(ransac_iterations):
#             # Choose random 4 point correspondences
#             indices = np.random.choice(len(matches), 4, replace=False)
#             src_subset = src_points[indices]
#             dst_subset = dst_points[indices]
#
#             # Compute homographic transformation from the 4 point correspondences
#             homography, _ = cv2.findHomography(src_subset, dst_subset, cv2.RANSAC, ransac_threshold)
#
#             # Calculate residuals for all correspondences
#             transformed_points = cv2.perspectiveTransform(src_points, homography)
#             residuals = np.linalg.norm(dst_points - transformed_points, axis=-1)
#
#             # Find inliers within threshold
#             inliers = residuals < ransac_threshold
#
#             # Update best homography and inliers if current model is better
#             if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
#                 best_homography = homography
#                 best_inliers = inliers
#
#     return best_affine

def RANSAC_Implementation(matches, kp1, kp2, trans_type):
    ransac_iterations = 10000
    ransac_threshold = 5
    src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # RANSAC loop
    best_transformation = None
    best_inliers = None
    if trans_type == "affine":
        for i in range(ransac_iterations):
            # Choose random 3 point correspondences
            indices = np.random.choice(len(matches), 3, replace=False)
            src_subset = src_points[indices]
            dst_subset = dst_points[indices]

            # Compute affine transformation from the 3 point correspondences
            affine_transform = solver(src_subset, dst_subset, "affine")

            # Calculate residuals for all correspondences
            transformed_points = cv2.transform(src_points, affine_transform)
            residuals = np.linalg.norm(dst_points - transformed_points, axis=-1)

            # Find inliers within threshold
            inliers = residuals < ransac_threshold

            # Update best transformation and inliers if current model is better
            if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
                best_transformation = affine_transform
                best_inliers = inliers
    else:
        for i in range(ransac_iterations):
            # Choose random 4 point correspondences
            indices = np.random.choice(len(matches), 4, replace=False)
            src_subset = src_points[indices]
            dst_subset = dst_points[indices]

            # Compute homographic transformation from the 4 point correspondences
            homography, _ = cv2.findHomography(src_subset, dst_subset, cv2.RANSAC, ransac_threshold)

            # Calculate residuals for all correspondences
            transformed_points = cv2.perspectiveTransform(src_points, homography)
            residuals = np.linalg.norm(dst_points - transformed_points, axis=-1)

            # Find inliers within threshold
            inliers = residuals < ransac_threshold

            # Update best transformation and inliers if current model is better
            if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
                best_transformation = homography
                best_inliers = inliers

    return best_transformation


# Find an existing implementation of the warping function, which can activate the transformation on an image preferably select
# Advanced interpolation option, such as bilinear or cubic-bi.
def warping(source_image, trans_matrix, type_, dim1, dim2):
    im = None
    if type_ == "affine":
        affine_mat = np.array([trans_matrix[0], trans_matrix[1]], dtype=np.float32)

        # inv_mat = cv2.invertAffineTransform(affine_mat)

        im = cv2.warpAffine(source_image, trans_matrix, (dim1, dim2), flags=cv2.INTER_LINEAR)

    else:

        im = cv2.warpPerspective(source_image, trans_matrix, (dim1, dim2), flags=cv2.INTER_LINEAR)

    return im


# Extract Dimensions for the full picture from .txt file.
def get_height_width(filename):
    match = re.match(r'.*?_H_(\d+)_W_(\d+)\..*', filename)
    if match:
        height = int(match.group(1))
        width = int(match.group(2))
        return height, width
    else:
        return None


def stitch_img(left, right, H):
    print("stiching image ...")

    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None,
                         0.0, 1.0, cv2.NORM_MINMAX)
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None,
                          0.0, 1.0, cv2.NORM_MINMAX)

    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape

    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    black = np.zeros(3)  # Black pixel.

    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass

    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image