import torch
import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import csv

class ClassificationDataset(Dataset):
    def __init__(self, path="", transform=None, category="train", useHFlip=False, useVFlip=False, num_images=None):
        self.path = path
        self.transform = transform
        self.category = category
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.num_images = num_images

        image_root_path = os.path.join(self.path, self.category)  # E.g., path/train or path/validation
        self.image_paths = []
        self.class_names = []
        self.class_to_idx = {}

        # Populate image paths and class information
        for class_idx, class_name in enumerate(sorted(os.listdir(image_root_path))):
            class_dir = os.path.join(image_root_path, class_name)
            if os.path.isdir(class_dir):
                image_files = sorted([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg','.JPEG'))])

                # If num_images is specified, limit the number of images selected from the subfolder
                if self.num_images is not None:
                    image_files = image_files[:self.num_images]

                self.image_paths.extend(image_files)
                self.class_names.append(class_name)
                self.class_to_idx[class_name] = class_idx

        self.total_images = len(self.image_paths)

    def __len__(self):
        return self.total_images

    def __getitem__(self, index):
        # Load image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB (handle grayscale images)

        # Get class label from the folder structure
        class_name = os.path.basename(os.path.dirname(image_path))
        class_idx = self.class_to_idx[class_name]
        # # Debugging: Print image path and class label
        # print(f"Image: {image_path}, Label: {class_idx}")
        # Flipping (if enabled)
        if self.useHFlip and random.random() > 0.5:
            image = ImageOps.mirror(image)  # Horizontal flip
        if self.useVFlip and random.random() > 0.5:
            image = ImageOps.flip(image)  # Vertical flip

        # Apply any provided transformations
        image = self.transform(image)
        image = image[[2, 1, 0], :, :]  # Swap channels to BGR
        # Return image and label (class index)
        return image, class_idx


def generate_csv(dataset, csv_filename):
    """Generates a CSV file with image paths and corresponding class labels."""
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_Path', 'Class_Label'])  # Write the header
        for idx in range(len(dataset)):
            image_path = dataset.image_paths[idx]
            class_label = dataset.class_names[dataset.class_to_idx[os.path.basename(os.path.dirname(image_path))]]
            writer.writerow([image_path, class_label])  # Write each image and label to the CSV
    print(f"CSV file saved as {csv_filename}")