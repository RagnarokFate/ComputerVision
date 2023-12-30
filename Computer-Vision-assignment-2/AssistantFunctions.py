import cv2
import numpy as np

def census_transform(image):

    # Get the image dimensions
    height, width = image.shape

    # Define the offsets for neighbor pixels
    offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i != 0 or j != 0)]

    # Initialize the Census Transform output
    census = np.zeros_like(image, dtype=np.uint8)

    # Perform the Census Transform
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Get the intensity value of the center pixel
            center_value = image[y, x]

            # Initialize the binary code
            binary_code = 0

            # Compare the center pixel with its neighbors
            for offset in offsets:
                neighbor_value = image[y + offset[0], x + offset[1]]
                binary_code <<= 1
                binary_code |= int(center_value > neighbor_value)

            # Assign the binary code to the transformed image
            census[y, x] = binary_code

    return census

def compute_volume_cost(left_image, right_image, max_disparity):
    height, width = left_image.shape[:2]
    volume_cost = np.zeros((height, width, max_disparity))

    for y in range(height):
        for x in range(width):
            for d in range(max_disparity):
                # Calculate the matching cost for assigning disparity d
                target_x = x - d
                if target_x < 0:
                    target_x = 0

                cost = np.abs(int(left_image[y, x]) - int(right_image[y, target_x]))

                # Assign the cost to the corresponding entry in the cost volume
                volume_cost[y, x, d] = cost

    return volume_cost

def compute_aggregate_local(volume_cost,window_size):

    height, width, max_disparity = volume_cost.shape
    aggregated_costs = np.zeros((height, width, max_disparity))

    # Define the weight matrix for uniform averaging within the window
    weight_matrix = np.ones((window_size, window_size, max_disparity))

    # Pad the cost volume to handle boundaries
    padded_cost_volume = np.pad(volume_cost, (
    (window_size // 2, window_size // 2), (window_size // 2, window_size // 2), (0, 0)), mode='constant')

    for y in range(height):
        for x in range(width):
            # Extract the local neighborhood from the padded cost volume
            neighborhood = padded_cost_volume[y:y + window_size, x:x + window_size, :]

            # Apply 2D weighted average to each layer of the local neighborhood
            aggregated_costs[y, x, :] = np.mean(neighborhood * weight_matrix, axis=(0, 1))


    return aggregated_costs