import cv2
import numpy as np
import os

def add_noise(img, variance = 0.05):
    
    noise = np.random.normal(0, variance, img.shape)
    noise = np.clip(noise, 0, 255)  # Clip noise values to 0-255
    noisy_img = img + noise.astype(np.uint8)  # Ensure noise is uint8 for addition

    return noisy_img


# Read the original image
img_ori = cv2.imread("image/LENNAorijinal.bmp", cv2.IMREAD_GRAYSCALE)

# Output directory
output_directory = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Add salt and pepper noise to the original image
img_noisy = add_noise(img_ori)

# Apply median filtering (MS2) with a 3x3 kernel
kernel_size = 3
img_filtered = cv2.medianBlur(img_noisy, kernel_size)

# Save the noisy and filtered images
cv2.imwrite("output/noisy.bmp", img_noisy)
cv2.imwrite("output/filtered.bmp", img_filtered)

# Display the noisy and filtered images
cv2.imshow("Noisy Image", img_noisy)
cv2.imshow("Filtered Image (MS2)", img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
