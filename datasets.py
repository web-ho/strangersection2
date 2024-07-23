#MIT License
#
#Copyright (c) 2024 Vaibhav Patel
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.




import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from utils import normalize_image


class StrangerSectionsDataset(Dataset):
    def __init__(self, root_dir, filenames_list, transforms=None, mode='train'):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, f'{mode}_images')
        self.label_folder = os.path.join(root_dir, f'{mode}_labels')
        self.image_filenames = filenames_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_id = os.path.splitext(self.image_filenames[idx])[0]
        img_name = os.path.join(self.image_folder, f'{image_id}.JPG')
        label_name = os.path.join(self.label_folder, f'{image_id}_gt.npy')

        image = np.asarray(Image.open(img_name).convert('RGB'))
        image = normalize_image(image, mode='imagenet')
        image = image.astype(np.float32) 

        mask = np.load(label_name)
        mask = mask.astype(np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
                        
        return image, mask.long()
    


class Inference_dataset(Dataset):
    def __init__(self, root_dir, image_filenames, transforms=None):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'image')
        self.image_filenames = image_filenames
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        image = np.asarray(Image.open(img_name).convert('RGB'))
        image = normalize_image(image, mode='imagenet')
        image = image.astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, img_name