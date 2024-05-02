import cv2
import numpy as np
import os
from scipy import ndimage, signal

# Read the original image
img_ori = cv2.imread("image/LENNAorijinal.png", cv2.IMREAD_GRAYSCALE)

# Output directory
output_directory = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def wiener_filter(img, noise_power):
    """
    Applies a Wiener filter to a grayscale image for noise reduction.

    Args:
        img (np.ndarray): Input grayscale image.
        noise_power (float): Estimated power of the noise.

    Returns:
        np.ndarray: Wiener-filtered grayscale image.
    """

    # Get image dimensions
    rows, cols = img.shape

    # Create frequency domain representation (shifted to center for FFT)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)  # Shift zero-frequency to the center

    # Assuming a blur filter as the degradation function (replace with actual PSF if known)
    # Consider using a more appropriate degradation function based on noise characteristics
    psf = np.ones((rows, cols)) / (rows * cols)

    # Estimate power spectrum of the original image (consider more robust power spectrum estimation)
    # Explore methods like Welch's method or periodogram averaging for improved estimation
    img_power_spectrum = np.abs(dft_shifted) ** 2

    # Wiener filter transfer function
    H_wien = np.conj(dft_shifted) / (img_power_spectrum + noise_power * psf)

    # Apply the filter
    filtered_dft = dft_shifted * H_wien

    # Shift back and perform inverse DFT
    filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_img = cv2.idft(filtered_dft_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)  # Clip to valid pixel range

    return filtered_img


def gaussian_filter(img, sigma):
    """
    Applies a Gaussian filter to a grayscale image for smoothing.

    Args:
        img (np.ndarray): Input grayscale image.
        sigma (float): Standard deviation of the Gaussian filter (controls blur strength).

    Returns:
        np.ndarray: Gaussian-filtered grayscale image.
    """

    return ndimage.gaussian_filter(img, sigma, mode='reflect')


# Display the original image
cv2.imshow('Original Image', img_ori)
cv2.waitKey(0)

# Apply Wiener filtering with estimated noise power
estimated_noise_power = 10  # Adjust this value based on your noise analysis
filtered_img_wien = wiener_filter(img_ori, estimated_noise_power)

# Display the Wiener-filtered image
cv2.imshow('Wiener Filtered', filtered_img_wien)

# Save the Wiener-filtered image
cv2.imwrite(output_directory + '/filtered_wien.png', filtered_img_wien)
print("Wiener filtering result saved to filtered_wien.png")
cv2.waitKey(0)

# Apply Gaussian filtering with different sigma values
sigma_values = [1, 2, 3]
for sigma in sigma_values:
    filtered_img_gauss = gaussian_filter(img_ori, sigma)
    cv2.imshow(f'Gaussian Filtered (sigma={sigma})', filtered_img_gauss)
    cv2.imwrite(output_directory + f'/filtered_gauss_{sigma}.png', filtered_img_gauss)
    print(f"Gaussian filtering result with sigma={sigma} saved to filtered_gauss_{sigma}.png")
    cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
