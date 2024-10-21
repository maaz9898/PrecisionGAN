# train.py
import os
import argparse
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from model import UnetGenerator, Discriminator
from utils import (
    Train_Normalize,
    Val_Normalize,
    weights_init,
    generator_loss,
    discriminator_loss,
)

torch.cuda.empty_cache()

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        version = 1
        new_output_dir = f"{output_dir}_{version}"
        while os.path.exists(new_output_dir):
            version += 1
            new_output_dir = f"{output_dir}_{version}"
        os.mkdir(new_output_dir)
        output_dir = new_output_dir
    return output_dir

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Create datasets and dataloaders
    train_ds = ImageFolder(
        args.train_dir, transform=transforms.Compose([Train_Normalize()])
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_ds = ImageFolder(
        args.val_dir, transform=transforms.Compose([Val_Normalize()])
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Model instantiation
    generator = (
        UnetGenerator(3, 3, 64, use_dropout=args.use_dropout)
        .apply(weights_init)
        .to(device)
    )
    summary(generator, (3, 256, 256))  # Call summary before DataParallel
    generator = torch.nn.DataParallel(generator)

    discriminator = (
        Discriminator(6, 64, n_layers=3)
        .apply(weights_init)
        .to(device)
    )
    summary(discriminator, (6, 256, 256))  # Call summary before DataParallel
    discriminator = torch.nn.DataParallel(discriminator)

    # Optimizers
    G_optimizer = optim.Adam(
        generator.parameters(),
        lr=args.gen_lr,
        betas=(0.5, 0.999),
    )
    D_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.dis_lr,
        betas=(0.5, 0.999),
    )

    # Create output directories
    output_dir = create_output_dir(args.output_dir)
    sub_folders = ["training_weights", "images"]
    for folder in sub_folders:
        folder_path = os.path.join(output_dir, folder)
        os.mkdir(folder_path)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=os.path.join("runs", output_dir))

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        epoch_folder = os.path.join(output_dir, f"images/epoch_{epoch}")
        if not os.path.exists(epoch_folder):
            os.mkdir(epoch_folder)

        D_loss_list, G_loss_list = [], []

        for (input_data, _label) in train_dl:
            input_img, target_img = input_data  # Unpack the input and target images
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            # Zero gradients for optimizers
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            # Generate images
            generated_image = generator(input_img)

            # Prepare inputs for discriminator
            disc_inp_fake = torch.cat((input_img, generated_image.detach()), 1)
            disc_inp_real = torch.cat((input_img, target_img), 1)
            real_target = torch.ones(
                input_img.size(0), 1, 30, 30, device=device
            )
            fake_target = torch.zeros(
                input_img.size(0), 1, 30, 30, device=device
            )

            # Train Discriminator
            D_fake = discriminator(disc_inp_fake)
            D_fake_loss = discriminator_loss(D_fake, fake_target)
            D_real = discriminator(disc_inp_real)
            D_real_loss = discriminator_loss(D_real, real_target)
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_total_loss.backward()
            D_optimizer.step()
            D_loss_list.append(D_total_loss.item())

            # Train Generator
            G_optimizer.zero_grad()
            fake_gen = torch.cat((input_img, generated_image), 1)
            G = discriminator(fake_gen)
            G_loss = generator_loss(generated_image, target_img, G, real_target)
            G_loss.backward()
            G_optimizer.step()
            G_loss_list.append(G_loss.item())

        # Log losses and images in TensorBoard
        avg_D_loss = sum(D_loss_list) / len(D_loss_list)
        avg_G_loss = sum(G_loss_list) / len(G_loss_list)
        writer.add_scalar("Loss/Discriminator", avg_D_loss, epoch)
        writer.add_scalar("Loss/Generator", avg_G_loss, epoch)

        print(
            f"Epoch [{epoch}/{args.num_epochs}]: D_loss: {avg_D_loss:.3f}, G_loss: {avg_G_loss:.3f}"
        )

        # Save sample images from validation set
        count = 0
        for (val_data, _label) in val_dl:
            val_input, _ = val_data  # Only input is needed for generation
            val_input = val_input.to(device)
            generated_output = generator(val_input)
            utils.save_image(
                generated_output.data[:100],
                os.path.join(epoch_folder, f"sample_{count}.png"),
                nrow=10,
                normalize=True,
            )
            count += 1
            break  # Only save one batch of images

        # Save models
        torch.save(
            generator.state_dict(),
            os.path.join(
                output_dir,
                f"training_weights/generator_epoch_{epoch}.pth",
            ),
        )

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Input batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=300,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--gen_lr",
        type=float,
        default=2e-4,
        help="Learning rate for generator",
    )
    parser.add_argument(
        "--dis_lr",
        type=float,
        default=2e-5,
        help="Learning rate for discriminator",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="torch",
        help="Directory to output training results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loader",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help='List of GPU IDs to use, e.g., "0,1"',
    )
    parser.add_argument(
        "--use_dropout",
        action="store_true",
        help="Use dropout in generator",
    )
    args = parser.parse_args()
    main(args)
