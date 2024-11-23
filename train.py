######################################## Export PATH Before running code######################
# export PYTHONPATH=$PYTHONPATH:/home/saqib/deeplearningresearch/python
##############################################################################################
import torch
import random
import datetime
from datetime import date
from dataloader.ClassificationDataset import ClassificationDataset, generate_csv
from utils.nf_helper import save_checkpoint, load_checkpoint, AGC
from model.NFNet import NFNet_v1
from tqdm import tqdm
import numpy as np
import datetime
from datetime import date
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.nn import DataParallel
from PIL import Image

####################### NFNet Trainer ###########################

def train(load_previous_model=True):
    # Device configuration
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"The device is: {device}")

    # Set random seeds for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if USE_CUDA:
        torch.cuda.manual_seed_all(777)

    # Hyperparameters
    imageWidth, imageHeight = 512, 512  # ImageNet image size
    batchSize = 768
    learningRate =  0.016* batchSize /4096
    epochs = 100
    targetAccuracy = 0.99999

    # Transformations
    transform = transforms.Compose([
    transforms.Resize((imageHeight, imageWidth), interpolation=Image.NEAREST),  # Bilinear interpolation for smoother resizing
    transforms.ToTensor(),  # Converts image to tensor with values in the range [0, 1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet mean/std
    ])

    # Load datasets
    trainDataset = ClassificationDataset(path="//home2/Read_only_Folder/ImageNet_Classification/", transform=transform, category="train",num_images=1)
    validDataset = ClassificationDataset(path="//home2/Read_only_Folder/ImageNet_Classification/", transform=transform, category="val", num_images=1)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=False)
    validLoader = DataLoader(validDataset, batch_size=1, shuffle=False, drop_last=False)

    TotalTrainBatch=len(trainLoader)
    # Initialize model
    num_classes = 1000  # ImageNet has 1000 classes
    model = NFNet_v1(class_num=num_classes)
    
    ######################### Using Multiple GPUs ############################
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    model = model.to(device)  
    ###########################################################################
    
    print('Total_batch_size=', TotalTrainBatch)
    # Model summary
    print('==== model info ====')
    summary(model, (3, imageHeight, imageWidth))
    print('====================')

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=learningRate)

        # Wrap the optimizer with AGC
    agc_optimizer = AGC(
        params=model.parameters(),
        optim=optimizer,
        clipping=1e-2,
        eps=1e-3,
        model=model.module if isinstance(model, DataParallel) else model,
        ignore_agc=["fc"]  
    )


    # Track best accuracy
    best_train_acc = 0.0
    best_model_path = None

    # Directory to save model
    Base_dir = '/home/saqib/deeplearningresearch/python/project/Pre_Training/Trained_Models/Classification'
    model_name = "NFNet_v1"
    build_date = str(date.today())
    model_dir = os.path.join(Base_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    generate_csv(trainDataset, os.path.join(model_dir, 'train_dataset.csv'))
    generate_csv(validDataset, os.path.join(model_dir, 'valid_dataset.csv'))

    existing_runs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join(model_dir, f"run_{run_number}")
    os.makedirs(run_dir)
    metrics = []

    # Check if a checkpoint exists
    checkpoint_dir=os.path.join(run_dir,'Checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize start_epoch to 0
    start_epoch = 0  # Default start
    # Only load the previous model if `load_previous_model` is True
    if load_previous_model:
        # Check if previous run exists to load from the latest checkpoint
        previous_run_number = run_number - 1
        previous_run_dir = os.path.join(model_dir, f"run_{previous_run_number}")
        previous_checkpoint_dir = os.path.join(previous_run_dir, 'Checkpoints')

        # Load the latest checkpoint from the previous run, if available
        if os.path.exists(previous_checkpoint_dir):
            checkpoint_files = sorted([f for f in os.listdir(previous_checkpoint_dir) if f.startswith('checkpoint_epoch_')])
            if checkpoint_files:
                last_checkpoint = checkpoint_files[-1]  # Get the latest checkpoint
                checkpoint_path = os.path.join(previous_checkpoint_dir, last_checkpoint)
                start_epoch = load_checkpoint(checkpoint_path, model, optimizer) + 1  # Start from next epoch
                print(f"Loaded checkpoint from {previous_checkpoint_dir} at epoch {start_epoch}")
            else:
                print("No checkpoint files found in the previous run.")
        else:
            print("No previous run found to load a checkpoint from.")
    else:
        print("Starting from scratch, not loading any previous model.")


    for epoch in range(start_epoch, epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        with tqdm(total=len(trainLoader), desc=f"Epoch {epoch}/{epochs-1}", unit="batch") as pbar:
            for images, labels in trainLoader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Backward pass and optimization
                agc_optimizer.zero_grad()
                
                loss.backward()
                agc_optimizer.step()

                # Update scheduler
                # scheduler.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)

                predicted = predicted.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                train_total += labels.shape[0]
                train_correct += (predicted == labels).sum().item()

                train_loss += loss.detach().cpu().numpy().item() 

                # current_lr = scheduler.get_last_lr()[0]

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), accuracy= train_correct / train_total)

        # Calculate mean loss and accuracy for the epoch
        train_loss /= len(trainLoader)
        train_acc = train_correct / train_total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in validLoader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                # Forward pass
                val_outputs = model(val_images).detach()
                val_loss_batch = loss_fn(val_outputs, val_labels)

                # Calculate accuracy
                _, val_predicted = val_outputs.max(1)

                val_predicted = val_predicted.detach().cpu().numpy()
                val_labels = val_labels.detach().cpu().numpy()

                val_total += val_labels.shape[0]
                val_correct += (val_predicted == val_labels).sum().item()

                val_loss += val_loss_batch.item()

        # Calculate mean validation loss and accuracy
        val_loss /= len(validLoader)
        val_acc = val_correct / val_total

        # Print results
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.9f}, Train Acc: {train_acc:.9f}, Val Loss: {val_loss:.9f}, Val Acc: {val_acc:.9f}')

        ################Save checkpoint after each epoch###################
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_checkpoint(checkpoint_state, model, checkpoint_dir, epoch)
        ####################################################################

        # Save best model based on validation accuracy
        if train_acc > best_train_acc:
            if best_model_path is not None:
                os.remove(best_model_path)  # Remove previous best model
            best_train_acc = train_acc
            best_model_path = os.path.join(run_dir, f"{build_date}_best_epoch_{epoch}_train_acc_{best_train_acc:.5f}_train_loss_{train_loss:.5f}_val_acc_{val_acc:.5f}_val_loss_{val_loss:.5f}.pth")
            State_dict_best_model_path = os.path.join(run_dir, f"state_dict_{build_date}_best_epoch_{epoch}_train_acc_{best_train_acc:.5f}_train_loss_{train_loss:.5f}_val_acc_{val_acc:.5f}_val_loss_{val_loss:.5f}.pth")

            torch.save(model, best_model_path)
            torch.save(model.state_dict(), State_dict_best_model_path)
            
            # # Save the scripted TorchScript model
            # scripted_model = torch.jit.script(model.module if isinstance(model, DataParallel) else model)
            # scripted_model.save(best_model_path)
            print(f"Saved best model at epoch {epoch} with training accuracy: {best_train_acc:.9f}")

        # Early stopping based on accuracy
        if val_acc >= targetAccuracy:
            print(f"Target accuracy reached at epoch {epoch }. Stopping training.")
            break



        # Store metrics
        metrics.append({
            'Epoch': epoch,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_acc
        })


        
        # Save and plot metrics every 10 epochs
        if (epoch) % 2 == 0 or epoch == epochs:
            # Save metrics to CSV every 10 epochs
            metrics_df = pd.DataFrame(metrics)
            csv_path = os.path.join(run_dir, f"training_metrics_epoch_{epoch}.csv")
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved training metrics to {csv_path}")

            # Plot metrics and save
            plot_metrics(metrics_df, run_dir, epoch)

    # Save final model
    final_model_path = os.path.join(run_dir, f"{build_date}_last_epoch_{epoch}_train_acc_{train_acc:.9f}_train_loss_{train_loss:.9f}_val_acc_{val_acc:.9f}_val_loss_{val_loss:.9f}.pth")
    torch.save(model, final_model_path)
    # Save the scripted TorchScript model
    # scripted_model = torch.jit.script(model.module if isinstance(model, DataParallel) else model)
    # scripted_model.save(final_model_path)
    print(f"Saved final model after epoch {epoch}")

def plot_metrics(metrics_df, run_dir, epoch):
    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Training Accuracy')
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy (Up to Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'accuracy_plot_epoch_{epoch}.png'))
    plt.close()

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Train Loss'], label='Training Loss')
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Up to Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'loss_plot_epoch_{epoch}.png'))
    plt.close()


if __name__ == '__main__':
    train()
