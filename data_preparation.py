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

from pseudo_data import CreatePseudoSamples



# function to pad images
def pad_image(pil_img, config):
    width, height = pil_img.size
    new_width = width + config.right + config.left
    new_height = height + config.top + config.bottom
    result = Image.new(pil_img.mode, (new_width, new_height), config.color)
    result.paste(pil_img, (config.left, config.top))
    return result


# function to pad labels
def pad_label(label, config):
    top = config.top
    bottom = config.bottom
    left = config.left
    right = config.right
    label = np.pad(label, ((top, bottom), (left, right)), 'constant', constant_values=0)
    return label


class CreatePatches():
    """
        Initialize the CreatePatches object.

        Args:
        - config (object): Configuration object containing patching parameters.
        - root_dir (str): Root directory containing images and labels.
        - file_list (list of str): List of filenames to process.
        - output_dir (str): Directory where patches will be saved.
        """
    def __init__(self, config, root_dir=None, file_list=None, output_dir=None):
        self.config = config
        self.file_ids = file_list
        self.image_folder = os.path.join(root_dir, 'image')
        self.label_folder = os.path.join(root_dir, 'label')
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.output_dir = output_dir
    
    # load image 
    def load_image(self, idx):
        filename = os.path.splitext(self.file_ids[idx])[0]
        file_path = os.path.join(self.image_folder, f'{filename}.JPG')
        if not os.path.exists(file_path):
            file_path = os.path.join(self.image_folder, f'{filename}.jpg')
        img = Image.open(file_path)
        image = np.asarray(pad_image(img, self.config))
        return image, filename
    
    # load mask
    def load_mask(self, idx):
        filename = os.path.splitext(self.file_ids[idx])[0]
        label = np.load(os.path.join(self.label_folder, f'{filename}_gt.npy'))
        label = pad_label(label, self.config)
        return label

    def generate_patches(self, array):
        """
        Generate patches from a given array based on patch size and stride.

        Args:
        - array (np.ndarray): Input array (image or label) to generate patches from.

        Returns:
        - patches (np.ndarray): Array containing generated patches.
        - total_patches (int): Total number of patches generated.
        """
        H, W = array.shape[:2]
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride
        num_patches_y = (H - patch_height) // stride_height + 1
        num_patches_x = (W - patch_width) // stride_width + 1

        patches = []
        for i in range(0, num_patches_y * stride_height, stride_height):
            for j in range(0, num_patches_x * stride_width, stride_width):
                patch = array[i:i + patch_height, j:j + patch_width]
                patches.append(patch)
        patches = np.array(patches)
        total_patches = (num_patches_x * num_patches_y)
        return patches, total_patches

    def create_all_patches(self, mode='train'):
        """
        Create and save all patches (images and labels) for a given mode.

        Args:
        - mode (str): Mode indicating whether patches are for training or validation.

        Returns:
        - patch_names (list of str): List of generated patch names.
        """
        label_patch_dir = os.path.join(self.output_dir, f'{self.config.exp_name}/{mode}_labels/')
        image_patch_dir = os.path.join(self.output_dir, f'{self.config.exp_name}/{mode}_images/')

        if not os.path.exists(label_patch_dir):
            os.makedirs(label_patch_dir, exist_ok=True)
            
        if not os.path.exists(image_patch_dir):
            os.makedirs(image_patch_dir, exist_ok=True)

        patch_names = []
        for idx in range(len(self.file_ids)):
            image, filename = self.load_image(idx)
            label = self.load_mask(idx)
            
            image_patches, total_patches = self.generate_patches(image)
            label_patches, _ = self.generate_patches(label)
            
            for i, patch in enumerate(image_patches):
                patch_name = f'{filename}_{i}'
                patch_image = Image.fromarray(patch)
                patch_filename = os.path.join(image_patch_dir, f"{filename}_{i}.JPG")
                patch_image.save(patch_filename)
                patch_names.append(patch_name)
            
            for i, patch in enumerate(label_patches):
                patch_filename = os.path.join(label_patch_dir, f"{filename}_{i}_gt.npy")
                np.save(patch_filename, patch)

        print(f"Number of patches created: {total_patches * len(self.file_ids)}")
        return patch_names
    


def prepare_data(config, train_dir=None, unlabeled_dir=None):
    """
    Prepare training and validation data lists based on the configuration.

    Args:
    - config (object): Configuration object containing dataset parameters.
    - train_dir (str): Directory containing training data.
    - unlabeled_dir (str): Directory containing unlabeled data.

    Returns:
    - train_list (list of str): List of filenames for training.
    - val_list (list of str): List of filenames for validation.
    """

    if not config.pseudo:
        images_list = sorted(os.listdir(os.path.join(train_dir, 'image')))
        val_list = [fname for fname in config.filenames if fname in images_list]
        train_list = [img for img in images_list if img not in val_list]
    else:
        CreatePseudoSamples(root_dir=unlabeled_dir, output_dir=train_dir)
        images_list = sorted(os.listdir(os.path.join(train_dir, 'image')))
        val_list = [fname for fname in config.filenames if fname in images_list]
        train_list = [img for img in images_list if img not in val_list]
    return train_list, val_list



def create_patches(config, train_dir=None, output_dir=None, train_list=None, val_list=None):
    """
    Create patches for training and validation datasets.

    Args:
    - config (object): Configuration object containing patching parameters.
    - train_dir (str): Directory containing training data.
    - output_dir (str): Directory where patches will be saved.
    - train_list (list of str): List of filenames for training.
    - val_list (list of str): List of filenames for validation.

    Returns:
    - train_patch_names (list of str): List of names of training patches created.
    - valid_patch_names (list of str): List of names of validation patches created.
    """

    train_patch_creator = CreatePatches(config, root_dir=train_dir, file_list=train_list, output_dir=output_dir)
    train_patch_names = train_patch_creator.create_all_patches(mode='train')

    valid_patch_creator = CreatePatches(config, root_dir=train_dir, file_list=val_list, output_dir=output_dir) 
    valid_patch_names = valid_patch_creator.create_all_patches(mode='val')
    
    return train_patch_names, valid_patch_names