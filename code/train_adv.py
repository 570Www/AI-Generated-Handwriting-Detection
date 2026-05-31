import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import csv

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from BADNet import BADNet

from HWDataset import HandwritingDataset_word
from discriminator import *


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def train_loop_adv(
    model, bcsd,
    train_loader,
    optimizer_D, optimizer_A,
    criterion,
    device, is_main, adv_weight
):
    model.train()
    bcsd.train()

    total_loss_D = 0.0
    total_loss_A = 0.0
    num_batches = 0

    iterator = train_loader
    if is_main:
        iterator = tqdm(train_loader, desc="Training (BCSD)", dynamic_ncols=True)

    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)

        # =====================================================
        # (1) Forward clean once to get semantic features
        # =====================================================
        with torch.no_grad():
            _ = model(images)

        feat4 = model.module.feat4
        feat5 = model.module.feat5
        feat6 = model.module.feat6
        # feat4 = model.module.feat4.detach()
        # feat5 = model.module.feat5.detach()
        # feat6 = model.module.feat6.detach()

        # =====================================================
        # (2) Generate hard images
        # =====================================================
        x_hard, delta = bcsd(images, feat4, feat5, feat6)
        # x_hard, delta = bcsd(images.clone(), feat4, feat5, feat6)

        # =====================================================
        # (3) Update discriminator
        # =====================================================
        optimizer_D.zero_grad()

        logits_clean = model(images)
        logits_hard = model(x_hard.detach())

        loss_D = (
            criterion(logits_clean, labels)
            + criterion(logits_hard, labels)
        )

        loss_D.backward()
        optimizer_D.step()

        # =====================================================
        # (4) Update BCSD-Net
        # =====================================================
        optimizer_A.zero_grad()

        logits_hard = model(x_hard)
        margin = (logits_hard[:, 1] - logits_hard[:, 0]).abs()

        m_min, m_max = 0.2, 1.0
        loss_margin = (
            F.relu(margin - m_max).pow(2)
            + F.relu(m_min - margin).pow(2)
        ).mean()

        with torch.no_grad():
            _ = model(images)
            feat5_clean = model.module.feat5.detach()

        _ = model(x_hard)
        feat5_hard = model.module.feat5

        loss_sem = F.mse_loss(feat5_hard, feat5_clean)
        loss_def = delta.pow(2).mean()

        loss_A = adv_weight * (loss_margin + 5.0 * loss_sem + 0.1 * loss_def)

        loss_A.backward()
        optimizer_A.step()

        # =====================================================
        # Logging
        # =====================================================
        total_loss_D += loss_D.item()
        total_loss_A += loss_A.item()
        num_batches += 1

        if is_main:
            iterator.set_postfix({
                "L_D": loss_D.item(),
                "L_A": loss_A.item()
            })

    avg_loss_D = total_loss_D / num_batches
    avg_loss_A = total_loss_A / num_batches

    return avg_loss_D, avg_loss_A



def validate_loop(model, val_loader, criterion, device, is_main):
    model.eval()
    total_loss = 0
    correct = 0
    N = 0

    iterator = val_loader
    if is_main:
        iterator = tqdm(val_loader, desc="Validating", dynamic_ncols=True)

    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            N += labels.size(0)

            if is_main:
                iterator.set_postfix({"val_loss": loss.item()})

    return total_loss / len(val_loader), correct / N


def save_to_csv(epoch, train_loss, val_loss, val_acc, epoch_time, csv_writer):
    if csv_writer is None:
        return
    csv_writer.writerow([epoch, train_loss, val_loss, val_acc, epoch_time])

def save_hyperparameters(args, output_dir):
    # Save hyperparameters (args) to super_parameters.csv
    super_params_path = os.path.join(output_dir, 'super_parameters.csv')
    header = ['Parameter', 'Value']
    
    # Check if file exists, if not create with header
    if not os.path.exists(super_params_path):
        with open(super_params_path, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)  # Write header
    
    # Append the hyperparameters
    with open(super_params_path, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        for arg, value in vars(args).items():
            csv_writer.writerow([arg, value])


def main():
    # ---------------------------------
    # Argument parsing (hyperparameters)
    # ---------------------------------
    parser = argparse.ArgumentParser(description="Train Handwriting Classifier")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default="./checkpoints/B/WordStylist", help='Directory to save CSV and models')
    parser.add_argument('--dataset_name', type=str, default="WordStylist")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    # ---------------------------------
    # Dataset + DDP samplers
    # ---------------------------------
    train_ds, val_ds, _ = HandwritingDataset_word.from_root("./experiments/data", data=args.dataset_name)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

    # ---------------------------------
    # Model
    # ---------------------------------
    model = DeiTDualHandwritingClassifier().to(device)

    for idx in [4, 5, 6]:
        for p in model.backbone.blocks[idx].parameters():
            p.requires_grad = True

    # for p in model.backbone.parameters():
    #     p.requires_grad = True
    # for p in model.encoder.parameters():
    #     p.requires_grad = False

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()

    optimizer_D = optim.Adam(
        list(model.module.classifier.parameters()) +
        list(model.module.backbone.blocks[4].parameters()) +
        list(model.module.backbone.blocks[5].parameters()) +
        list(model.module.backbone.blocks[6].parameters()),
        lr=args.learning_rate
    )
    # optimizer_D = optim.Adam(
    #     model.module.parameters(),
    #     lr=args.learning_rate
    # )

    bcsd = BADNet().to(device)
    bcsd = DDP(bcsd, device_ids=[local_rank], find_unused_parameters=True)

    optimizer_A = optim.Adam(
        bcsd.parameters(),
        lr=1e-4
    )

    # Prepare CSV output file
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_file = os.path.join(args.output_dir, 'training_log.csv')
        with open(csv_file, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Acc', 'Epoch Time'])
        save_hyperparameters(args, args.output_dir)
    
    # ---------------------------------
    # Training loop
    # ---------------------------------
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n===== Epoch {epoch} =====")

        # Train phase
        adv_weight = min(1.0, (epoch + 1) / 10.0)
        start_time = time.time()
        train_loss_D, train_loss_A = train_loop_adv(
            model, bcsd,
            train_loader,
            optimizer_D, optimizer_A,
            criterion,
            device, is_main, adv_weight
        )
        train_time = time.time() - start_time

        # Validation phase
        start_time = time.time()
        val_loss, val_acc = validate_loop(model, val_loader, criterion, device, is_main)
        val_time = time.time() - start_time

        # Log the results and save to CSV
        if is_main:
            print(f"[Epoch {epoch}] "
                f"Train L_D: {train_loss_D:.4f}, "
                f"Train L_A: {train_loss_A:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"Train Time: {train_time:.2f}s, "
                f"Val Time: {val_time:.2f}s")

            # Save to CSV
            with open(csv_file, mode='a', newline='') as f:
                csv_writer = csv.writer(f)
                save_to_csv(epoch, train_loss_D, val_loss, val_acc, train_time + val_time, csv_writer)

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.output_dir, f"discriminator_epoch_{epoch+1}.pth")
                # torch.save({
                #     "cnn": model.module.backbone.state_dict(),
                #     "classifier": model.module.classifier.state_dict(),
                # }, save_path)
                torch.save({
                    "epoch": epoch + 1,
                    "discriminator": model.module.state_dict(),
                    "bcsd": bcsd.module.state_dict(),
                }, save_path)
                print(f"[Checkpoint] Saved model to {save_path}")


    cleanup_ddp()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_adv.py