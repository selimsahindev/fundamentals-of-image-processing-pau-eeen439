import os
import cv2
import numpy as np

# Output directory
output_directory = 'output/color_filter'

# Create output folder if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the image
image = cv2.imread('image/peppers.png')

# Create versions with only one color channel
blue = cv2.merge([image[:, :, 0], np.zeros_like(image[:, :, 0]), np.zeros_like(image[:, :, 0])])
green = cv2.merge([np.zeros_like(image[:, :, 1]), image[:, :, 1], np.zeros_like(image[:, :, 1])])
red = cv2.merge([np.zeros_like(image[:, :, 2]), np.zeros_like(image[:, :, 2]), image[:, :, 2]])

# Save the images
cv2.imwrite(output_directory + '/peppers_blue_filter.png', blue)
cv2.imwrite(output_directory + '/peppers_green_filter.png', green)
cv2.imwrite(output_directory + '/peppers_red_filter.png', red)

# Display the images horizontally
final = np.hstack((blue, green, red))
cv2.imshow('Color Filter', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
