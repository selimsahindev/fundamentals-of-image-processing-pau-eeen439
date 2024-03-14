import os
import numpy as np
import cv2

def rgb_to_hsi(rgb):
    """
    Converts an RGB image to HSI color space.

    Parameters:
    rgb (numpy.ndarray): The input RGB image.

    Returns:
    numpy.ndarray: The HSI image.
    """
    # Extract the R, G, and B channels
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # Calculate the intensity
    intensity = (r + g + b) / 3

    # Calculate the saturation
    minimum = np.minimum(np.minimum(r, g), b)
    saturation = 1 - (3 / (r + g + b)) * minimum

    # Calculate the hue
    hue = np.arccos((0.5 * ((r - g) + (r - b))) / (np.sqrt((r - g)**2 + (r - b) * (g - b) + 1e-6)))

    # Set the hue to 0 if B > G
    hue[b > g] = 2 * np.pi - hue[b > g]

    # Convert the hue to degrees
    hue = hue * 180 / np.pi

    # Stack the HSI channels
    hsi = np.dstack((hue, saturation, intensity))

    return hsi


# Load your image
img = cv2.imread("image/peppers.png")

# Output directory
output_directory = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Convert to HSI colorspace
hsi_img = rgb_to_hsi(img)

# Save the image
cv2.imwrite(output_directory + "/hsi_filter.png", hsi_img)

# Display the image
cv2.imshow("HSI Image", hsi_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
