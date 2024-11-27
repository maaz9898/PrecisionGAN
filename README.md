# PrecisionGAN: 

## Enhanced Image-to-Image Translation for Preserving Structural Integrity in Skeletonized Images

PrecisionGAN is a deep learning-based framework designed for skeletonizing characters in ancient manuscripts, emphasizing the preservation of structural integrity. This repository contains scripts for training, testing, and post-processing, leveraging a U-Net-based generator with multi-head attention and a PatchGAN discriminator for effective image-to-image translation.

---

## **Features**

- U-Net Generator with multi-head attention and residual connections.
- Custom PatchGAN Discriminator with spectral normalization and dropout.
- Robust training pipeline with support for multi-GPU training via PyTorch's `DataParallel`.
- Post-processing tools for refining skeletonized images, including morphological operations.
- Versatile model applicable beyond skeletonization, such as underwater image restoration.

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/maaz9898/PrecisionGAN.git
cd PrecisionGAN
```

### **2. Set Up the Virtual Environment**

It’s recommended to use a virtual environment for managing dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/MacOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### **3. Install Dependencies**

Install all the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Note:** Ensure you install the correct version of PyTorch for your system. You can check the [PyTorch installation page](https://pytorch.org/get-started/locally/) to download the appropriate version for your CUDA setup.

---

## **Usage**

### **1. Training the Model**

To train the model on a dataset, use the following command:

```bash
python train.py --train_dir <path_to_train_dataset> --val_dir <path_to_val_dataset> --batch_size 64 --num_epochs 500 --gen_lr 0.0002 --dis_lr 0.00002 --output_dir <output_directory>
```

#### **Example:**

```bash
python train.py --train_dir Dataset/train --val_dir Dataset/val --batch_size 64 --num_epochs 500 --gen_lr 0.0002 --dis_lr 0.00002 --output_dir result
```

#### **Arguments:**

- `--train_dir`: Path to the training dataset.
- `--val_dir`: Path to the validation dataset.
- `--batch_size`: Number of images per batch during training.
- `--num_epochs`: Number of epochs to train the model.
- `--gen_lr`: Learning rate for the generator.
- `--dis_lr`: Learning rate for the discriminator.
- `--output_dir`: Directory to save training results and models.

### **2. Testing the Model**

To test a trained model on a dataset, use the following command:

```bash
python test.py --input_folder <path_to_test_images> --output_folder <output_directory> --model_weights <path_to_pretrained_weights> --gpu_ids <GPU_IDs>
```

#### **Example:**

```bash
python test.py --input_folder testing_data --output_folder testing_output --model_weights pretrained_weights.pth --gpu_ids 0
```

#### **Arguments:**

- `--input_folder`: Path to the folder containing input test images.
- `--output_folder`: Path to save the output generated images.
- `--model_weights`: Path to the pre-trained model weights.
- `--gpu_ids`: Comma-separated list of GPU IDs to use.

### **3. Post-Processing**

To apply post-processing techniques like thinning and removing isolated pixels, use the following command:

```bash
python post_process.py --input_folder <path_to_input_images> --output_folder <path_to_output_images>
```

#### **Example:**

```bash
python post_process.py --input_folder testing_output --output_folder testing_pp
```

#### **Arguments:**

- `--input_folder`: Path to the folder containing images to post-process.
- `--output_folder`: Path to save the post-processed images.

---

## **File Structure**

```text
├── train.py               # Training script
├── test.py                # Inference script
├── post_process.py        # Post-processing script
├── model.py               # U-Net Generator and Discriminator architecture
├── utils.py               # Utility functions for data preprocessing and augmentation
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## **Requirements**

- Python 3.8+
- PyTorch 2.0.1+
- TorchVision 0.15.2+
- NumPy
- OpenCV
- Pillow
- SciPy
- Matplotlib
- TensorBoard

Install the required packages using `pip install -r requirements.txt`.

---

## **Citation**

If you use this project in your research, please cite it using the following:

```bibtex
@article{ahmed2024precisiongan,
  title={PrecisionGAN: enhanced image-to-image translation for preserving structural integrity in skeletonized images},
  author={Ahmed, Maaz and Kim, Min-Beom and Choi, Kang-Sun},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  pages={1--15},
  year={2024},
  publisher={Springer}
}
```

---
