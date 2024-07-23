import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


exp_name = 'maxxvit_exp_11'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 29
    
epochs = 30     #30
batch_size = 16
num_workers = 0
in_chans = 3
img_size_h = 256
img_size_w = 256

lr = 1e-4
optimizer = 'Adam'
scheduler = 'CosineAnnealingLR'
scheduler1 = 'ReduceLROnPlateau'
min_lr = 9e-5
patience = 5

patch_size = (224, 224)
stride = (224, 224)
normalize = 'imagenet'
pseudo = True


train_transform = A.Compose([
    A.Resize(img_size_h, img_size_w, interpolation=cv2.INTER_CUBIC),
    ToTensorV2(),
    ])

test_transform = A.Compose([
    A.Resize(img_size_h, img_size_w, interpolation=cv2.INTER_CUBIC),
    ToTensorV2(),
    ])

filenames = ['3gpbh5.JPG', '45zs2b.JPG', '79h0ma.JPG', 'tpb83i.JPG', 'tya5k0.JPG', 'v2rf63.JPG', 'vutdxm.JPG', 'wc3ks0.JPG', 'wo91nj.JPG', 'ynfeq0.JPG', 'ypbf6w.JPG', 'zjl4vx.JPG']


top = 48
right = 104 
bottom = 48
left = 104
color = (0,0,0)


ENCODER = 'tu-maxxvit_rmlp_small_rw_256.sw_in1k'
ENCODER_WEIGHTS = 'imagenet' 
ENCODER_DEPTH = 5
DECODER_CHANNELS = (256, 128, 64, 32, 16)
CLASSES = 4
ACTIVATION = None 
DECODER_ATTENTION_TYPE = None


###### Inference ########

test_batch_size = 1
roi_size = (256,256) 
sw_batch_size = 1 
overlap = 0.7  
mode = "constant"  
padding_mode = "reflect" 