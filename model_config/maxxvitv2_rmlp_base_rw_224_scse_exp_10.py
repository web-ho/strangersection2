import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


exp_name = 'maxxvitv2_exp_10'

seed = 29
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 30       #30
batch_size = 16
num_workers = 0
in_chans = 3
img_size_h = 224
img_size_w = 224

lr = 1e-4
optimizer = 'Adam'
scheduler = 'CosineAnnealingLR'
scheduler1 = 'ReduceLROnPlateau'
min_lr = 1e-5
patience = 5

patch_size = (224, 224)
stride = (224, 224)
normalize = 'imagenet'
pseudo = True


train_transform = A.Compose([
        ToTensorV2(),
    ])

test_transform = A.Compose([
        ToTensorV2(),
    ])

filenames = ['79h0ma.JPG', 'tpb83i.JPG', 'vutdxm.JPG', 'wo91nj.JPG', 'ynfeq0.JPG', 'ypbf6w.JPG', 'zjl4vx.JPG']


top = 48
right = 104 
bottom = 48
left = 104
color = (0,0,0)


ENCODER = 'tu-maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k'
ENCODER_WEIGHTS = 'imagenet' 
ENCODER_DEPTH = 5
DECODER_CHANNELS = (256, 128, 64, 32, 16)
CLASSES = 4
ACTIVATION = None 
DECODER_ATTENTION_TYPE = 'scse'


###### Inference ########

test_batch_size = 1
roi_size = (224,224) 
sw_batch_size = 1 
overlap = 0.7  
mode = "constant"  
padding_mode = "reflect" 