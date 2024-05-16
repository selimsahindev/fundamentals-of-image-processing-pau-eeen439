import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # Import for directory handling

def calculate_pixel_wise_variance(image, window_size):
  """
  Calculates the pixel-wise variance of a grayscale image.

  Args:
    image: The grayscale image as a NumPy array.
    window_size: The size of the neighborhood window for variance calculation.

  Returns:
    A NumPy array containing the pixel-wise variance values.
  """

  variance = np.zeros_like(image)
  for i in range(window_size // 2, image.shape[0] - window_size // 2):
    for j in range(window_size // 2, image.shape[1] - window_size // 2):
      # Check for at least two valid values in the window
      if np.count_nonzero(image[i - window_size // 2:i + window_size // 2 + 1,
                                 j - window_size // 2:j + window_size // 2 + 1]) >= 2:
        window = image[i - window_size // 2:i + window_size // 2 + 1,
                       j - window_size // 2:j + window_size // 2 + 1]
        variance[i, j] = np.var(window)
      else:
        # Assign default value (e.g., 0) for edge pixels
        variance[i, j] = 0
  return variance


def save_image(image, filename, output_dir):
  """
  Saves an image to the specified output directory.

  Args:
    image: The image data to save.
    filename: The filename for the saved image.
    output_dir: The directory path to save the image.
  """
  cv2.imwrite(os.path.join(output_dir, filename), image)


# Specify image path and output directory
image_path = "image/kugu.jpg"
output_dir = "output"

# Read the image
image = cv2.imread(image_path)

# Check if output directory exists, create it if not
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding using Otsu's algorithm
_, thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the grayscale and thresholded images (optional)
cv2.imshow("Grayscale Image", grayscale_image)
cv2.imshow("Thresholded Image", thresholded_image)

# Define window size for variance calculation
window_size = 3

# Calculate pixel-wise variance
variance = calculate_pixel_wise_variance(grayscale_image, window_size)

# Normalize the variance values
normalized_variance = variance / np.max(variance)

# Plot the pixel-wise variance and save it
plt.imshow(normalized_variance, cmap='viridis')
plt.title("Pixel-wise Variance")
plt.colorbar()
plt.savefig(os.path.join(output_dir, "variance.png"))
plt.show()

# Calculate the histogram
hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

# Plot the histogram and save it
plt.plot(hist)
plt.xlabel("Grayscale Level")
plt.ylabel("Pixel Count")
plt.title("Histogram")
plt.savefig(os.path.join(output_dir, "histogram.png"))
plt.show()

# Save grayscale and thresholded images
save_image(grayscale_image, "grayscale.jpg", output_dir)
save_image(thresholded_image, "thresholded.jpg", output_dir)

# Wait for a keypress to close windows
cv2.waitKey(0)
cv2.destroyAllWindows()  # Explicitly close OpenCV windows
