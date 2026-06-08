import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PairedEUVPDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.low_quality_path = os.path.join(root_dir, "trainA" if mode == "train" else "validation")  
        self.high_quality_path = os.path.join(root_dir, "trainB") if mode == "train" else None  

        # Saves the list of LQ image filenames & ensures alignment with HQ image names (by order)
        self.transform = transform
        self.low_quality_images = sorted(os.listdir(self.low_quality_path))

        # Ensures that both LQ and HQ sets have equal lengths (truncates extra if needed).
        if self.high_quality_path:
            self.high_quality_images = sorted(os.listdir(self.high_quality_path))
            min_length = min(len(self.low_quality_images), len(self.high_quality_images))  
            self.low_quality_images = self.low_quality_images[:min_length]
            self.high_quality_images = self.high_quality_images[:min_length]

    def __len__(self):
        return len(self.low_quality_images) #total no. of samples

    def __getitem__(self, idx): # Loads a low-quality image at the given index and converts it to RGB.
        low_img = Image.open(os.path.join(self.low_quality_path, self.low_quality_images[idx])).convert("RGB")

        # Loads corresponding HQ image - Applies transformations (e.g., .ToTensor()) - Returns a pair: (low_img, high_img).
        if self.high_quality_path:
            high_img = Image.open(os.path.join(self.high_quality_path, self.high_quality_images[idx])).convert("RGB")
            if self.transform:
                low_img = self.transform(low_img)
                high_img = self.transform(high_img)
            return low_img, high_img  
        else:
            if self.transform: #  for validation 
                low_img = self.transform(low_img)
            return low_img  

# **✅ Updated Transformations (NO Resize, Keeps 240x320)**
transform = transforms.Compose([
    transforms.ToTensor(),
])

# **✅ Define dataset path**
dataset_path = "/kaggle/input/euvp-paired-underwater-scenes/underwater_scenes"

# **✅ Load train and validation sets**
train_dataset = PairedEUVPDataset(dataset_path, mode="train", transform=transform)
val_dataset = PairedEUVPDataset(dataset_path, mode="validation", transform=transform)

# **✅ Set num_workers for Kaggle compatibility** no parallel processing
num_workers = 0 if "KAGGLE_KERNEL_RUN_MODE" in os.environ else min(2, os.cpu_count()) 

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)

# **✅ Debugging Info**
print(f"✅ Training set size: {len(train_dataset)} images")
print(f"✅ Validation set size: {len(val_dataset)} images")
print(f"✅ Number of train batches: {len(train_loader)}")
print(f"✅ Number of validation batches: {len(val_loader)}")
print(f"🔍 First 5 LQ images: {train_dataset.low_quality_images[:5]}")
if train_dataset.high_quality_path:
    print(f"🔍 First 5 HQ images: {train_dataset.high_quality_images[:5]}")
