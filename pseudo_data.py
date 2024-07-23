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
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.inferers import SlidingWindowInferer
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import Inference_dataset
from model import CreateModel
from utils import *


# manually selected images from the unlabeled images directory to generate pseudo labels 
pseudo_img_names = [
    '0a6odx.jpg', '0fsbjg.jpg', '0gcvnh.jpg', '5ouia3.jpg', '5r1qui.jpg', '5wco2i.jpg', '5x38mn.jpg', '5z28sl.jpg', '5ze8xt.jpg', '65krxf.jpg', 
    '6akwqp.jpg', '6cf7vb.jpg', '79n3vx.jpg', '7yk0sh.jpg', '81iehw.jpg', '8407zu.jpg', 'amsz4h.jpg', 'b6qgut.jpg', 'bw10u9.jpg', 'bw1qyi.jpg', 
    'bxo4zc.jpg', 'c2qmod.jpg', 'cquik6.jpg', 'iaj6lv.jpg', 'kring4.jpg', 'l7hrj6.jpg', 'lf54cb.jpg', 'lkq4r6.jpg', 'ltbgxs.jpg', 'qjtp93.jpg', 
    'qs13fa.jpg', 'qwu26t.jpg', 'rn6815.jpg', 's73m2a.jpg', 'sb2tvx.jpg', 't341lj.jpg', 'y6edrs.jpg', 'yqpo4e.jpg', 'yv49u3.jpg', 'zew2st.jpg', 
    'zmq807.jpg', 'zo93fs.jpg'
]


def save_pseudodata(filenames, preds, output_dir, source):
    """
    Save pseudo data (predicted labels and corresponding images) to specified directories.

    Args:
    - filenames (list of str): List of filenames for the original images.
    - preds (list of np.ndarray): List of predicted label arrays.
    - output_dir (str): Directory where pseudo data will be saved.
    - source (str): Directory containing original images as the source to copy from.
    """

    label_dir = os.path.join(output_dir, 'label')
    image_dir = os.path.join(output_dir, 'image')

    if not os.path.exists(label_dir):
        os.makedirs(label_dir, exist_ok=True)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    for filename, pred in zip(filenames, preds):
        img_id = os.path.splitext(os.path.basename(filename))[0]
        pseudo_label_name = os.path.join(label_dir, f"{img_id}_gt.npy")
        np.save(pseudo_label_name, pred)
        
        src_file = os.path.join(source, f"{img_id}.jpg")
        dst_file = os.path.join(image_dir, f"{img_id}.jpg")
        
        if not os.path.exists(dst_file):
            shutil.copy(src_file, dst_file)




def generate_pseudo_labels(config, unlabeled_dir):
    """
    Generate pseudo labels for unlabeled images using a trained model.

    Args:
    - config (object): Configuration object containing various parameters.
    - unlabeled_dir (str): Directory containing unlabeled images and labels.

    Returns:
    - pseudo_preds (list of np.ndarray): List of predicted pseudo labels as numpy arrays.
    - filenames (list of str): List of filenames corresponding to the pseudo labels.
    """

    unlabelled_images_list = sorted(os.listdir(os.path.join(unlabeled_dir, 'image')))
    pseudo_img_list = [fname for fname in pseudo_img_names if fname in unlabelled_images_list]
    
    transform = A.Compose([ToTensorV2()])
    inference_dataset = Inference_dataset(unlabeled_dir, pseudo_img_list[:3], transform)
    inference_loader = DataLoader(inference_dataset, batch_size=config.test_batch_size, num_workers=config.num_workers, shuffle=False)

    model = CreateModel(config)
    model_path = f"weights/unetplusplus_{config.exp_name}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(config.device)
    
    inferer = SlidingWindowInferer(roi_size=config.roi_size, sw_batch_size=config.sw_batch_size, overlap=config.overlap, mode=config.mode, padding_mode=config.padding_mode)

    pseudo_preds = []
    filenames = []
    with torch.no_grad():
        for image, file_paths in tqdm(inference_loader):
            image = image.to(config.device)
            outputs = inferer(inputs=image, network=model)
            
            predicted_masks = torch.argmax(outputs, dim=1)
            pseudo_preds.extend(predicted_masks.cpu().numpy())
            filenames.extend(file_paths)
    return pseudo_preds, filenames


def CreatePseudoSamples(configs_dict, root_dir=None, output_dir=None):
    """
    Generate pseudo samples using multiple model configurations explicitly defined in exp_names list and save them.

    Args:
    - configs_dict (dict): Dictionary containing model configurations.
    - root_dir (str): Root directory containing unlabeled images and labels.
    - output_dir (str): Directory where pseudo samples will be saved.
    """

    exp_names = [
        'maxxvit_exp_1',
        'maxxvitv2_exp_2',
        'maxxvitv2_exp_3',
        'maxxvitv2_exp_4',
        'maxvit_exp_5',
        'maxvit_exp_6'
    ]

    all_model_preds = []

    for model_config in exp_names:
        if model_config in configs_dict:
            cfg = configs_dict[model_config]
            print(f"Processing model config: {model_config}")
            pseudo_preds, filenames = generate_pseudo_labels(cfg, root_dir)
            all_model_preds.append(pseudo_preds)
        else:
            raise ValueError(f"Model config '{model_config}' not found in model_configs_dict")
    
    # Ensemble predictions from all model configurations
    final_pseudo_preds = ensemble_predictions(all_model_preds)
    # Clear predictions to reduce noise
    cleared_preds = clear_predictions(final_pseudo_preds)
    
    source = os.path.join(root_dir, 'image')
    save_pseudodata(filenames, cleared_preds, output_dir, source)
    torch.cuda.empty_cache()
