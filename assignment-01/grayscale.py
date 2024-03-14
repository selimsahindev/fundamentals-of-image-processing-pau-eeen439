import os
import cv2
import numpy as np

# Output directory
output_directory = 'output/grayscale'

# Create output folder if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the image
image = cv2.imread('image/peppers.png')

# Split the image into its channels
blue, green, red = cv2.split(image)

# Save the images
cv2.imwrite(output_directory + '/peppers_red.png', red)
cv2.imwrite(output_directory + '/peppers_green.png', green)
cv2.imwrite(output_directory + '/peppers_blue.png', blue)

# Create a grid of images
# Arrange images horizontally
grid = np.hstack((red, green, blue))

# Display the result
cv2.imshow('Red   Green   Blue', grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
