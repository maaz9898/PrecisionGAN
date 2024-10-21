# test.py
import os
import argparse
import cv2
import numpy as np
import torch
from model import UnetGenerator
from utils import IMG_WIDTH, IMG_HEIGHT

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Inference using Trained Generator")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input images directory",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output images directory",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to the trained generator model weights (.pth file)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help='Comma-separated GPU IDs to use (e.g., "0,1"). Use "-1" for CPU.',
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=".png,.jpg,.jpeg,.tiff,.bmp,.gif",
        help="Comma-separated list of image file extensions to process",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=IMG_WIDTH,
        help="Width to resize images for the model input",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=IMG_HEIGHT,
        help="Height to resize images for the model input",
    )
    return parser.parse_args()

def load_generator(model_weights, device):
    generator = UnetGenerator(3, 3, 64, use_dropout=False).to(device)
    state_dict = torch.load(model_weights, map_location=device)

    # Remove "module." prefix if present (from DataParallel)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    generator.load_state_dict(new_state_dict)
    generator.eval()
    return generator

def inference(generator, input_image_path, output_image_path, device, img_width, img_height):
    # Load and preprocess the input image
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Warning: Unable to read image {input_image_path}. Skipping.")
        return
    if len(input_image.shape) == 2 or input_image.shape[2] == 1:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    input_image = cv2.resize(input_image, (img_width, img_height), interpolation=cv2.INTER_AREA)
    input_image = input_image.transpose((2, 0, 1))
    input_image = (input_image.astype(np.float32) / 127.5) - 1  # Normalize to [-1, 1]
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        generated_image = generator(input_tensor)

    generated_image = generated_image.squeeze(0).cpu().numpy()
    generated_image = np.transpose(generated_image, (1, 2, 0))
    generated_image = (generated_image + 1) * 127.5  # Denormalize to [0, 255]
    generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_image_path, generated_image)

def process_folder(generator, input_folder, output_folder, device, image_extensions, img_width, img_height):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of valid image extensions
    valid_extensions = tuple(ext.strip().lower() for ext in image_extensions.split(','))

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                input_image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_image_path = os.path.join(output_dir, file)
                inference(generator, input_image_path, output_image_path, device, img_width, img_height)

def main():
    args = parse_arguments()

    # Set device
    if args.gpu_ids != "-1" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        device = torch.device("cuda")
        print(f"Using GPU IDs: {args.gpu_ids}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load the generator model
    generator = load_generator(args.model_weights, device)
    print("Generator model loaded successfully.")

    # Process images
    process_folder(
        generator,
        args.input_folder,
        args.output_folder,
        device,
        args.image_extensions,
        args.img_width,
        args.img_height,
    )
    print("Image processing completed.")

if __name__ == "__main__":
    main()
