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




from tqdm import tqdm
import torch
from sklearn.metrics import jaccard_score
from monai.inferers import SlidingWindowInferer


def train(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Function to train a PyTorch model.

    Args:
    - model (torch.nn.Module): The PyTorch model to train.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
    - criterion (torch.nn.Module): Loss function criterion.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - device (str): Device to run the training on ('cuda' or 'cpu').
    - scheduler (optional, torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.

    Returns:
    - train_loss (float): Average loss per batch during training.
    """
        
    model.train()  
    model.to(device)
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()  

    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch[0]
        masks = batch[1]
        images = images.to(device)
        masks = masks.to(device)
        
        # Automatic mixed precision training context
        with torch.cuda.amp.autocast(): 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward() 
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
    train_loss = running_loss / len(dataloader)

    return train_loss



def evaluate(model, dataloader, criterion, device, scheduler=None):
    """
    Function to evaluate a PyTorch model on a validation set.

    Args:
    - model (torch.nn.Module): The PyTorch model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing batches of validation data.
    - criterion (torch.nn.Module): Loss function criterion.
    - device (str): Device to run the evaluation on ('cuda' or 'cpu').
    - scheduler (optional, torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.

    Returns:
    - results (dict): Dictionary containing evaluation results.
      Contains 'loss' (list of float): Average loss per batch during evaluation.
      Contains 'jaccard' (list of float): Average Jaccard index score over all classes.
    """

    results = {
        'loss': [],
        'jaccard': [],
    }

    model.eval()
    model.to(device)
    running_loss = 0
    jaccard = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            images = batch[0]
            masks = batch[1]
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # Compute Jaccard score
            predicted_masks = torch.argmax(outputs, dim=1)
            true_masks = masks.cpu().numpy().reshape(-1)
            pred_masks = predicted_masks.cpu().numpy().reshape(-1)
            jaccard += jaccard_score(true_masks, pred_masks, average="micro", labels=[1,2,3])

        val_loss = running_loss / len(dataloader)
        jaccard /= len(dataloader)
        results['loss'].append(val_loss)
        results['jaccard'].append(jaccard)
        if scheduler is not None:
            scheduler.step(val_loss)

    return results
