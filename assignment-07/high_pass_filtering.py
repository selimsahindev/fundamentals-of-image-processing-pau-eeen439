import cv2
import numpy as np
import os
from scipy import ndimage, signal

# Read the original image
img_ori = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

# Output directory
output_directory = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def ideal_high_pass_filter(img, cutoff_frequency):
    """
    Applies an ideal high pass filter to a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image.
        cutoff_frequency (float): Cutoff frequency (normalized to half the image width). Values between 0 and 0.5 are valid.

    Returns:
        np.ndarray: Filtered grayscale image.
    """
    # Get image dimensions
    rows, cols = img.shape

    # Create frequency domain representation (shifted to center for FFT)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)  # Shift zero-frequency to the center

    # Create ideal high pass filter mask (consider normalization for better results)
    mask = np.zeros(dft_shifted.shape, dtype=np.float32)
    center_row, center_col = int(rows / 2), int(cols / 2)
    radius = int(cutoff_frequency * cols)  # Normalized radius based on cutoff frequency

    # Create circular mask with radius (consider normalization for better results)
    cv2.circle(mask, (center_col, center_row), radius, (1, 1), -1)  # Fill the circle (all frequencies within)

    # Apply the filter (consider normalization for better results)
    filtered_dft = dft_shifted * mask

    # Shift back and perform inverse DFT
    filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_img = cv2.idft(filtered_dft_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)  # Clip to valid pixel range

    return filtered_img


def butterworth_high_pass_filter(img, cutoff_frequency, order):
    """
    Applies a Butterworth high pass filter to a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image.
        cutoff_frequency (float): Cutoff frequency (normalized to half the image width). Values between 0 and 0.5 are valid.
        order (int): Order of the Butterworth filter (higher order provides steeper roll-off).

    Returns:
        np.ndarray: Filtered grayscale image.
    """

    # Get image dimensions
    rows, cols = img.shape

    # Create frequency domain representation (shifted to center for FFT)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)  # Shift zero-frequency to the center

    # Create Butterworth filter mask
    wn = cutoff_frequency  # Normalized cutoff frequency
    ny = 0.5  # Normalized half-image width

    # Use scipy.signal.butter to generate Butterworth filter coefficients (normalized)
    b, a = signal.butter(order, wn, btype='highpass')
    # Create the filter mask from the coefficients (consider normalization for better results)
    filter_response = np.abs(signal.freqz(b, a, worN=rows * cols)[1])
    filter_mask = filter_response.reshape([rows, cols])

    # Apply the filter (consider normalization for better results)
    filtered_dft = dft_shifted * filter_mask

    # Shift back and perform inverse DFT
    filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_img = cv2.idft(filtered_dft_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)  # Clip to valid pixel range

    return filtered_img


def gaussian_high_pass_filter(img, cutoff_frequency):
    """
    Applies a Gaussian high pass filter to a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image.
        cutoff_frequency (float): Standard deviation of the Gaussian filter (controls the width of the transition band). Higher values lead to a wider transition band.

    Returns:
        np.ndarray: Filtered grayscale image.
    """

    # Get image dimensions
    rows, cols = img.shape

    # Create frequency domain representation (shifted to center for FFT)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)  # Shift zero-frequency to the center

    # Create Gaussian high pass filter mask
    # Use scipy.ndimage.gaussian_filter to generate a 2D Gaussian filter
    gaussian_filter = ndimage.gaussian_filter([[1]], cutoff_frequency)  # 2D Gaussian with center at (0, 0)
    # Invert the Gaussian to create a high pass filter (consider normalization for better results)
    filter_mask = 1 - gaussian_filter

    # Apply the filter (consider normalization for better results)
    filtered_dft = dft_shifted * filter_mask

    # Shift back and perform inverse DFT
    filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_img = cv2.idft(filtered_dft_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)  # Clip to valid pixel range

    return filtered_img



cutoff_frequencies = [0.1, 0.2, 0.3, 0.5, 0.7]

# Display the original image and the filtered images for different cutoff frequencies in a 2x3 grid, in single window.
num_rows, num_cols = 2, 3  # Grid dimensions

# Create a window to display the images
cv2.namedWindow('High Pass Filtering', cv2.WINDOW_NORMAL)
cv2.resizeWindow('High Pass Filtering', 800, 600)

# Display the original image
cv2.imshow('High Pass Filtering', img_ori)
cv2.waitKey(0)

# Apply ideal high pass filter for different cutoff frequencies
for i, cutoff_frequency in enumerate(cutoff_frequencies):
    filtered_img = ideal_high_pass_filter(img_ori, cutoff_frequency)
    cv2.imshow('High Pass Filtering', filtered_img)
    cv2.imwrite(output_directory + f'/filtered_{cutoff_frequency}_ideal.png', filtered_img)
    cv2.waitKey(0)

# Apply Butterworth high pass filter for different cutoff frequencies and orders
for i, cutoff_frequency in enumerate(cutoff_frequencies):
    for j, order in enumerate([1, 2, 3]):
        filtered_img = butterworth_high_pass_filter(img_ori, cutoff_frequency, order)
        cv2.imshow('High Pass Filtering', filtered_img)
        cv2.imwrite(output_directory + f'/filtered_{cutoff_frequency}_{order}_butterworth.png', filtered_img)
        cv2.waitKey(0)

# Apply Gaussian high pass filter for different cutoff frequencies
for i, cutoff_frequency in enumerate(cutoff_frequencies):
    filtered_img = gaussian_high_pass_filter(img_ori, cutoff_frequency)
    cv2.imshow('High Pass Filtering', filtered_img)
    cv2.imwrite(output_directory + f'/filtered_{cutoff_frequency}_gaussian.png', filtered_img)
    cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
