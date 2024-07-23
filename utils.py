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
import random
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# normalization function
def normalize_image(image, mode=None):
    image = image / 255.0
    if mode == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = image.mean(axis=(0, 1), keepdims=True)
        std = image.std(axis=(0, 1), keepdims=True)
    normalized_image = (image - mean) / std
    return normalized_image


def clear_predictions(mask_images):
    """
    Clear predicted masks by replacing non-zero values with the dominant class label.

    Args:
    - mask_images (list of np.ndarray): List of predicted mask images as numpy arrays.

    Returns:
    - clear_masks (list of np.ndarray): List of cleared mask images where non-zero values are replaced
      with the dominant class label.
    """
    
    clear_masks = []
    for im in mask_images:
        im_copy = im.copy()
        # Count occurrences of each class label (excluding background)
        class_counts = np.bincount(im_copy.flatten())[1:]
        dominant_class = np.argmax(class_counts) + 1
        non_zero_mask = im_copy != 0
        im_copy[non_zero_mask] = dominant_class
        clear_masks.append(im_copy)
    return clear_masks


def ensemble_predictions(preds_list, num_classes=4):
    """
    Ensemble predictions from multiple models by averaging one-hot encoded masks.

    Args:
    - preds_list (list of list of np.ndarray): List of lists containing predicted mask images
      as numpy arrays from multiple models.
    - num_classes (int): Number of classes in the segmentation task (default is 4).

    Returns:
    - final_preds (list of np.ndarray): List of final ensemble predictions as numpy arrays.
    """

    final_preds = []
    for preds in zip(*preds_list):
        # Convert each predicted mask to one-hot encoding
        one_hot_masks = [F.one_hot(torch.from_numpy(pred).long(), num_classes) for pred in preds]
        # Stack one-hot masks along the channel dimension
        ensemble_mask = torch.stack(one_hot_masks, dim=2).float()
        averaged_mask = torch.mean(ensemble_mask, dim=2)
        final_pred = torch.argmax(averaged_mask, dim=2).numpy()
        final_preds.append(final_pred)
    return final_preds
    
        

def save_log(log, epoch, time, train_loss, val_loss, valid_jaccard):
    new_log = {
        "epoch": [epoch],
        "time": [time],
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "valid_jaccard": [valid_jaccard]
    }

    new_log = pd.DataFrame.from_dict(new_log)
    if log is not None:
        return pd.concat([log, new_log]).reset_index(drop=True)
    else:
        return new_log
    
    
def plot_loss(train_loss, val_loss, save_path=False):
    epochs = range(1, len(train_loss) + 1)
    train_loss = np.array(train_loss).flatten()
    val_loss = np.array(val_loss).flatten()

    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_jaccard(jaccard, save_path=False):
    epochs = range(1, len(jaccard) + 1)
    jaccard = np.array(jaccard).flatten()

    plt.plot(epochs, jaccard, 'g', label='Jaccard Score')
    plt.title('Jaccard Score')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Score')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()