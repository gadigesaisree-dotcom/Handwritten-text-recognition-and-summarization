import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def get_training_transforms():
    """Returns a set of data augmentations to improve HTR accuracy."""
    return T.Compose([
        T.RandomRotation(degrees=3),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05))
    ])

class IAMDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, processor=None, is_trocr=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample (mostly for CRNN).
            processor: HuggingFace processor (for TrOCR).
            is_trocr (bool): Whether the dataset is used for TrOCR (returns pixel_values instead of raw images).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.processor = processor
        self.is_trocr = is_trocr

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

       
        orig_path = self.data_frame.iloc[idx, 0]
    
        path_parts = orig_path.replace("\\", "/").split("/")
        try:
            data_idx = path_parts.index("data")
            rel_path = os.path.join(*path_parts[data_idx:])
        except ValueError:
            # Fallback if 'data' is not in path
            rel_path = os.path.basename(orig_path)
            
        img_name = os.path.join(self.root_dir, rel_path)
        
        text = str(self.data_frame.iloc[idx, 1])

        if self.is_trocr and self.processor is not None:
            try:
                image = Image.open(img_name).convert("RGB")
                
                if self.transform:
                    image = self.transform(image)
                    
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
                labels = self.processor.tokenizer(text, padding="max_length", max_length=128).input_ids
                labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
                return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}
            except Exception as e:
                return None
        else:
            image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            image = cv2.resize(image, (512, 128))
            image = image.astype('float32') / 255.0
            image = torch.from_numpy(image).unsqueeze(0) 
            
            sample = {'image': image, 'text': text}

            if self.transform:
                sample = self.transform(sample)

            return sample
