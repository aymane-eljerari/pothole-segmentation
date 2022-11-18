import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


# Link to dataset: https://sites.google.com/view/pothole-600/dataset
class SegDataset(Dataset):
    '''Pothole Segmentation Dataset dataset.'''

    def __init__(self, path):
        '''
        Args:
            path                (string):                       directory containing all images
            feature_exctractor  (SegformerFeatureExtractor):    Segformer input encoder from Hugging Face
        '''
        self.path               = path

        self.data_path          = os.path.join(self.path, 'rgb')
        self.label_path         = os.path.join(self.path, 'label')

        self.data               = torch.zeros((600, 3, 400, 400))
        self.label              = torch.zeros((600, 1, 400, 400))

        self.data_augment       = torch.zeros((7800, 13, 3, 400, 400))
        self.label_augment      = torch.zeros((7800, 13, 1, 400, 400))

        self.angles             = [-90, -60, -45, -30, -15, 15, 30, 45, 60, 90]

        self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_augment[idx], self.label_augment[idx]

    
    def generate_data(self):
        #TODO: dont forget random seed initialization for data transforms
        '''
        Generates and stores the data and labels
        '''
        # Generate Initial Dataset
        for i in range(600):
            img             = Image.open(os.path.join(self.data_path, str(i+1) + '.png'))
            label           = Image.open(os.path.join(self.label_path, str(i+601) + '.png')).convert('L')

            transform       = transforms.Compose([transforms.PILToTensor()])
            img_tensor      = torch.tensor(transform(img)).detach().clone().squeeze_()
            label_tensor    = torch.tensor(transform(label)).detach().clone().squeeze_()

            self.data[i]    = img_tensor
            self.label[i]   = label_tensor
        
        # Data Augmentation

        hflip               = transforms.RandomHorizontalFlip(p=1)
        vflip               = transforms.RandomVerticalFlip(p=1)

        self.data_hflip     = torch.stack([hflip(i) for i in self.data])
        self.label_hflip    = torch.stack([hflip(i) for i in self.label])

        self.data_vflip     = torch.stack([vflip(i) for i in self.data])
        self.label_vflip    = torch.stack([vflip(i) for i in self.label])

        self.data_rotation  = torch.stack([self.angle_rotation(i) for i in self.data])
        self.label_rotation = torch.stack([self.angle_rotation(i) for i in self.label])

        # Concatenate new dataset
        self.data_augment   = torch.cat((self.data, self.data_hflip, self.data_vflip, self.data_rotation))
        self.label_augment  = torch.cat((self.label, self.label_hflip, self.label_vflip, self.label_rotation))


    def angle_rotation(self, image):
        '''
            Args:
                    image       (Tensor):   input image as tensor
        '''
        images = torch.empty(size=(600*len(self.angles), TF.get_image_num_channels(image), 400, 400))
        count = 0
        for angle in range(len(self.angles)):
            img = TF.rotate(image, angle)
            images[count] = img
            count+=1
        return images


if __name__ == '__main__':

    dataset = SegDataset(path='data/segmentation/raw')

    print(dataset[0])
            




    