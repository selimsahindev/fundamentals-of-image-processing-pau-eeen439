import cv2
import numpy as np
import os

# Read the original image
img_ori = cv2.imread("image/LENNAorijinal.bmp", cv2.IMREAD_GRAYSCALE)

# Output directory
output_directory = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get the SNR value in dB
snr = 10

# Calculate the standard deviation of the noise
sigma = np.sqrt((10 ** (0.1 * snr)) / (1 - 10 ** (0.1 * snr)))

# Generate Gaussian noise
noise = np.random.normal(0, sigma, img_ori.shape)

# Add noise to the original image
img_noisy = img_ori + noise

# Save the noisy image
cv2.imwrite("output/LENNA_noisy.bmp", img_noisy)

# Define low-pass filters
kernel_3x3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
kernel_5x5 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]) / 25
kernel_15x15 = np.array([[1 for _ in range(15)] for _ in range(15)]) / 225

# Apply the filters
img_filtered_3x3 = cv2.filter2D(img_noisy, -1, kernel_3x3)
img_filtered_5x5 = cv2.filter2D(img_noisy, -1, kernel_5x5)
img_filtered_15x15 = cv2.filter2D(img_noisy, -1, kernel_15x15)

# Save the filtered images
cv2.imwrite("output/LENNA_filtered_3x3.bmp", img_filtered_3x3)
cv2.imwrite("output/LENNA_filtered_5x5.bmp", img_filtered_5x5)
cv2.imwrite("output/LENNA_filtered_15x15.bmp", img_filtered_15x15)

# Show the noisy image
cv2.imshow("Noisy", img_noisy)

# Show the filtered images
cv2.imshow("3x3", img_filtered_3x3)
cv2.imshow("5x5", img_filtered_5x5)
cv2.imshow("15x15", img_filtered_15x15)
cv2.waitKey(0)
