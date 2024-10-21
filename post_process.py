# post_process.py
import os
import argparse
import cv2
import numpy as np
from PIL import Image
import scipy.signal

def binarize_image(input_image_path, threshold=50):
    """
    Binarize an image by thresholding.

    Parameters:
    - input_image_path (str): Path to the input image.
    - threshold (int): Threshold value for binarization.

    Returns:
    - PIL.Image.Image: Binarized image.
    """
    # Open the input image
    image = Image.open(input_image_path)
    
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Check if the image is grayscale
    if len(image_np.shape) == 2:
        # The image is grayscale
        mask = (image_np > threshold)
    else:
        # The image is RGB
        mask = (image_np[:, :, 0] > threshold) | (image_np[:, :, 1] > threshold) | (image_np[:, :, 2] > threshold)

    # Set the pixels above the threshold to white (255), and below or equal to the threshold to black (0)
    image_np[mask] = 255
    image_np[~mask] = 0  # Ensure that the rest of the pixels are set to black
    
    # Convert the numpy array back to an image
    new_image = Image.fromarray(image_np.astype('uint8'))
    
    return new_image

def remove_isolated_pixels(binary_image, min_size=8):
    """
    Remove isolated pixels smaller than a minimum size.

    Parameters:
    - binary_image (numpy.ndarray): Binary image array.
    - min_size (int): Minimum size of connected components to keep.

    Returns:
    - numpy.ndarray: Image with small isolated pixels removed.
    """
    # Remove isolated pixels
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    sizes = stats[:, -1]

    new_img = np.zeros(binary_image.shape, np.uint8)
    for i in range(1, nb_components):
        if sizes[i] >= min_size:
            new_img[output == i] = 255
    return new_img

def fix_small_holes(binary_image, kernel_size=3):
    """
    Fix small holes in a binary image.

    Parameters:
    - binary_image (numpy.ndarray): Binary image array.
    - kernel_size (int): Size of the kernel for convolution.

    Returns:
    - numpy.ndarray: Image with small holes filled.
    """
    fixed_image = np.copy(binary_image)

    # Define the convolution kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=int)
    center = kernel_size // 2
    kernel[center, center] = 0  # Exclude the center pixel

    # Perform a 2D convolution
    convolved = scipy.signal.convolve2d(binary_image, kernel, mode='same', boundary='fill', fillvalue=0)

    # Threshold the convolved image to identify potential hole pixels
    threshold = kernel.sum() - 1
    hole_candidates = (convolved >= threshold)

    # Fill the identified holes in the fixed image
    fixed_image[hole_candidates] = 255  # Assuming holes are black (0) surrounded by white (255)

    return fixed_image

def thinning(binary_image):
    """
    Apply morphological thinning to a binary image.

    Parameters:
    - binary_image (numpy.ndarray): Binary image array.

    Returns:
    - numpy.ndarray: Thinned image.
    """
    # Apply the thinning operation
    thinned_image = cv2.ximgproc.thinning(binary_image)
    return thinned_image

def process_image(input_image_path, output_image_path, threshold, min_size, kernel_size):
    """
    Process a single image: binarize, remove isolated pixels, fix small holes, and thin.

    Parameters:
    - input_image_path (str): Path to the input image.
    - output_image_path (str): Path to save the processed image.
    - threshold (int): Threshold value for binarization.
    - min_size (int): Minimum size of connected components to keep.
    - kernel_size (int): Kernel size for hole fixing.
    """
    # Binarize the image
    img_pil = binarize_image(input_image_path, threshold=threshold)
    img_np = np.array(img_pil)  # Convert PIL Image to NumPy array

    # Convert to grayscale if necessary
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Ensure binary image
    _, img_binary = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # Remove isolated pixels
    img_processed = remove_isolated_pixels(img_binary, min_size=min_size)

    # Fix small holes
    img_processed = fix_small_holes(img_processed, kernel_size=kernel_size)

    # Apply thinning
    img_processed = thinning(img_processed)

    # Save the processed image
    cv2.imwrite(output_image_path, img_processed)

def process_folder(input_folder, output_folder, threshold, min_size, kernel_size):
    """
    Recursively process all images in a folder.

    Parameters:
    - input_folder (str): Path to the input directory.
    - output_folder (str): Path to the output directory.
    - threshold (int): Threshold value for binarization.
    - min_size (int): Minimum size of connected components to keep.
    - kernel_size (int): Kernel size for hole fixing.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for entry in os.scandir(input_folder):
        if entry.is_dir():
            # Recurse into subdirectories
            process_folder(
                entry.path,
                os.path.join(output_folder, entry.name),
                threshold,
                min_size,
                kernel_size,
            )
        else:
            # Process files
            filename = entry.name
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                input_image_path = entry.path
                output_image_path = os.path.join(output_folder, filename)
                process_image(input_image_path, output_image_path, threshold, min_size, kernel_size)

def main():
    parser = argparse.ArgumentParser(description="Post-process images by binarizing and thinning.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input images directory.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output images directory.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Threshold value for binarization (default: 50).",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=8,
        help="Minimum size of connected components to keep (default: 8).",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size for hole fixing (default: 3).",
    )
    args = parser.parse_args()

    # Check if OpenCV extra modules are available for thinning
    if not hasattr(cv2, 'ximgproc'):
        raise ImportError("OpenCV ximgproc module is required for thinning. Please install opencv-contrib-python.")

    # Process the folder
    process_folder(
        args.input_folder,
        args.output_folder,
        args.threshold,
        args.min_size,
        args.kernel_size,
    )
    print("Image processing completed.")

if __name__ == "__main__":
    main()
