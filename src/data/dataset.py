import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import SegformerFeatureExtractor

# Link to dataset: https://sites.google.com/view/pothole-600/dataset
class SegDataset(Dataset):
    '''Pothole Segmentation Dataset dataset.'''

    def __init__(self, path, feature_extractor):
        '''
        Args:
            
            parh                (string): directory containing all images
            feature_exctractor  (SegformerFeatureExtractor): Segformer input encoder from Hugging Face
        '''
        self.path               = path
        self.data_path          = os.path.join(self.path, 'rgb')
        self.label_path         = os.path.join(self.path, 'label')
        self.data               = torch.zeros((600, 3, 400, 400))
        self.labels             = torch.zeros((600, 1, 400, 400))
        self.feature_extractor  = feature_extractor

        self.generate_data()

    
    def generate_data(self):
        '''
        Generates and stores the data and labels
        '''
        for i in range(600):
            img             = cv2.imread(os.path.join(self.data_path, str(i+1) + '.png'))

            segment_image   = cv2.imread(os.path.join(self.label_path, str(i+601) + '.png'))
            segment_image   = cv2.cvtColor(segment_image, cv2.COLOR_RGB2GRAY)

            encoded_input   = self.feature_extractor(img, segment_image, return_tensors="pt")

            self.data[i]    = encoded_input['pixel_values'].squeeze()
            self.labels[i]  = encoded_input['labels'].squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == '__main__':

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", size=(400, 400))
    dataset = SegDataset(path='data/segmentation/raw', feature_extractor=feature_extractor)

    print(dataset[0])
            




    