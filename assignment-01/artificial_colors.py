import os
import cv2
import numpy as np

# Output directory
output_directory = 'output/artificial_colors'

# Create grayscale folder if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
# Read the image
image = cv2.imread('image/peppers.png')


# Split the image into its channels
blue, green, red = cv2.split(image)

# GRB
grb = cv2.merge([green, red, blue])
# RBG
rbg = cv2.merge([red, blue, green])
# GBR
gbr = cv2.merge([green, blue, red])

# Save the images
cv2.imwrite(output_directory + '/peppers_grb.png', grb)
cv2.imwrite(output_directory + '/peppers_rbg.png', rbg)
cv2.imwrite(output_directory + '/peppers_gbr.png', gbr)

# Display the images horizontally
final = np.hstack((grb, rbg, gbr))
cv2.imshow('Artificial Colors', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
