import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
import gc
import numpy as np
import wandb
from torch.utils.data import DataLoader

import random
import cv2

import torch
import torch.nn as nn
from torch.optim import AdamW

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from resnetall import generate_model
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.login(key='574c23ca47bbd885a816336238f90cdee9d3a9ca')

def Fifth_aug(img, strip_height=150):
    height, _ = img.shape[:2]
    kernel = np.ones((3, 3), np.uint8) 
    section_h = height//3

    # S2 texture
    img[section_h:section_h*2,:] = cv2.GaussianBlur(img[section_h:section_h*2,:],(15,15),0)

    # S3 texture
    img[section_h*2:,:] = cv2.erode(img[section_h*2:,:], kernel, iterations=2) 

    # S4 texture
    #num_strips = section_h // strip_height
    #for j in range(num_strips):
        #y1 = j * strip_height
        #y2 = min(y1 + strip_height, height)
        #strip = img[(section_h*3+y1):(section_h*3+y2), :]
        #if  j%3==0:
        #    strip = cv2.GaussianBlur(strip, (55, 55), 0)
        #    strip = cv2.multiply(strip, 0.25)
        #    strip = cv2.erode(strip, kernel, iterations=3) 

        #img[(section_h*3+y1):(section_h*3+y2), :] = strip

    return img


class CFG:
    # ============== comp exp name =============
    trial_no = 17
    exp_no = 4
    TRAIN_ON_DIFF = True
    TRAIN_SOBEL = False
    comp_name = f'Vesuvius_S2_T{trial_no}'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./' #         
    exp_name = f'Training_S2_T{trial_no}_E{exp_no}' # 
    fragments =     ["S1_20230929220920",       "S4_20231210132040",         "S3_20240618142020",       "S3_20240716140050",       "S3_20240712074250",       "S1_20231004222109",         'S1_20230515162442',        '20230517214715',        '20230422011040',       '20230422213203',       '20230426114804',       '20230522182853',       '20230709155141',       '20230801194757',          '20240516205750',   '20230509173534']
    start_idxs =    {"S1_20230929220920":17,    "S4_20231210132040":17,      "S3_20240618142020":17,    "S3_20240716140050":17,    "S3_20240712074250":17,    "S1_20231004222109":17,      'S1_20230515162442':17,      '20230517214715':17,    '20230422011040':17,    '20230422213203':17,    '20230426114804':17,    '20230522182853':17,    '20230709155141':17,    '20230801194757':17,    '20240516205750':17,    '20230509173534':17}
    reversed_load = {"S1_20230929220920":False, "S4_20231210132040":False,   "S3_20240618142020":False, "S3_20240716140050":False, "S3_20240712074250":False, "S1_20231004222109":False,   'S1_20230515162442':False,   '20230517214715':False, '20230422011040':False, '20230422213203':False, '20230426114804':False, '20230522182853':False, '20230709155141':False, '20230801194757':False, '20240516205750':False, '20230509173534':False}
    flipped =       {"S1_20230929220920":False, "S4_20231210132040":False,   "S3_20240618142020":False, "S3_20240716140050":False, "S3_20240712074250":False, "S1_20231004222109":False,   'S1_20230515162442':False,   '20230517214715':False, '20230422011040':False, '20230422213203':False, '20230426114804':False, '20230522182853':False, '20230709155141':False, '20230801194757':False, '20240516205750':False,' 20230509173534':False}

    # ================== model cfg =================
    ckpt = None
    backbone = 'resnet3d'
    in_chans = 26

    # ================== training cfg ==============
    size = 64
    tile_size = size * 4
    stride = tile_size // 8

    train_batch_size = 600
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 30

    # adamW warmupあり
    warmup_factor = 10
    lr = 1e-4

    # ================== val =======================
    valid_id = 'S1_20230515162442'

    # ================== fixed =====================
    print_freq = 40
    num_workers = 2
    seed = 42

    # ============== set dataset path ==============
    outputs_path = f'./ckpts/{comp_name}/{exp_name}/'
    model_dir = outputs_path + f'{comp_name}-models/'

    # ============== augmentation ==================
    Fourth = False
    Fifth = True

    if Fourth:print("Using Fourth")
    if Fifth:print("Using Fifth")

    CLAHE = A.Compose([A.CLAHE(clip_limit=(1,16), tile_grid_size=(4,4), p=1)])
    train_aug_list = [
        A.Resize(size, size),
        A.Morphological(p=0.125,  scale=(int(size * 0.0625), int(size * 0.0625)), operation='dilation'),
        A.Morphological(p=0.25,   scale=(int(size * 0.0625), int(size * 0.0625)), operation='erosion'),
        A.Morphological(p=0.125,  scale=(int(size * 0.0625), int(size * 0.0625)), operation='dilation'),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.35, scale_limit=0.35, p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.35),
        A.CoarseDropout(max_holes=3, max_width=int(size * 0.15), max_height=int(size * 0.15), 
                        mask_fill_value=0, p=0.5),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    
    rotate = A.Compose([A.Rotate(5, p=1)])


def set_seed(seed=42, cudnn_deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    pl.seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    os.makedirs(cfg.model_dir, exist_ok=True)
        

def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train': make_dirs(cfg)
       

def read_image_mask(fragment_id, start_idx=17, end_idx=43):
    # Read The labels and mask first.
    labels = cv2.imread( f"./segments/{fragment_id}/{fragment_id}_T{CFG.exp_no}_inklabels.png", 0)
    fragment_mask = cv2.imread(CFG.comp_dataset_path + f"./segments/{fragment_id}/{fragment_id}_mask.png", 0)
    assert fragment_mask.shape == labels.shape, "Labels shape must match the shape of the mask"

    # Read the layers.
    images = []
    if CFG.reversed_load[fragment_id]:
      print("Reversed Load")
      idxs = range(64, 64-CFG.in_chans, -1)
    else:
      idxs = range(start_idx, end_idx)


    print(list(idxs))
    if CFG.TRAIN_ON_DIFF:
      print("TRAINING ON DIFFs")
      for i in idxs:
          if CFG.Fifth and fragment_id != CFG.valid_id:
            if i != start_idx:
                #image1n = image0
                image0  = image1p
                image1p = Fifth_aug(cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+1):02}.tif", 0)).astype(np.float32)   

            else:
                #image1n = Fifth_aug(cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i-1):02}.tif", 0)).astype(np.float32)
                image0  = Fifth_aug(cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i  ):02}.tif", 0)).astype(np.float32)
                image1p = Fifth_aug(cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+1):02}.tif", 0)).astype(np.float32)

          else:
            if i != start_idx:
                #image1n = image0
                image0  = image1p
                image1p = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+1):02}.tif", 0).astype(np.float32)  

            else:
                #image1n = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i-1):02}.tif", 0).astype(np.float32)
                image0  = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i  ):02}.tif", 0).astype(np.float32)
                image1p = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+1):02}.tif", 0).astype(np.float32)

        #   if CFG.TRAIN_SOBEL:
        #     print("TRAINING ON SOBEL DIFFs")
        #     gX = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        #     gY = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        #     image1 = (gX**2 + gY**2)**0.5

        #     gX = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        #     gY = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        #     image2 = (gX**2 + gY**2)**0.5

          #dn = abs(image1n - image0)
          dp = abs(image0 - image1p)
          image = dp #(dn + dp)/2
          image = image.astype(np.uint8)
          #image = CFG.CLAHE(image=image)['image']

          pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
          pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

          image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)        
          images.append(image)

    else: 
      for i in idxs:
          image = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{i:02}.tif", 0)

          pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
          pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

          image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)        
          image = np.clip(image, 0, 200)
          images.append(image)

    images = np.stack(images, axis=2)
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    labels = labels.astype('float32')
    labels /= 255

    return images, labels, fragment_mask


def get_train_valid_dataset(fragment_ids):
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in fragment_ids:
        if not os.path.exists(f"./segments/{fragment_id}/{fragment_id}_T{CFG.exp_no}_inklabels.png"):
          print(f"No Labels found for {fragment_id}")
          print("*"*100)
          continue
          
        print('Reading:', fragment_id)
        image, mask, fragment_mask = read_image_mask(fragment_id, CFG.start_idxs[fragment_id], CFG.start_idxs[fragment_id] + CFG.in_chans)
        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
        windows_dict = {}

        for a in y1_list:
            for b in x1_list:
                for yi in range(0, CFG.tile_size, CFG.size):
                    for xi in range(0, CFG.tile_size, CFG.size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+CFG.size
                        x2=x1+CFG.size
                        if fragment_id!=CFG.valid_id:
                            if not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.05):
                                if not np.any(fragment_mask[a:a+ CFG.tile_size, b:b + CFG.tile_size]==0):
                                    train_images.append(image[y1:y2, x1:x2])
                                    train_masks.append(mask[y1:y2, x1:x2, None])
                                    assert image[y1:y2, x1:x2].shape==(CFG.size, CFG.size, CFG.in_chans)

                        if fragment_id==CFG.valid_id:
                            if (y1,y2,x1,x2) not in windows_dict:
                                if not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0):
                                        valid_images.append(image[y1:y2, x1:x2])
                                        valid_masks.append(mask[y1:y2, x1:x2, None])
                                        valid_xyxys.append([x1, y1, x2, y2])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size, CFG.size, CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'
        gc.collect()
        print("*"*100)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)

    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys = xyxys
        self.rotate = CFG.rotate

    def __len__(self):
        return len(self.images)
    
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(18, 26)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0

        image = image_tmp

        return image

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy = self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask'].unsqueeze(0)

            return image, label, xy
        
        else:
            image = self.images[idx]
            label = self.labels[idx]
            if CFG.Fourth:
                image = self.fourth_augment(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask'].unsqueeze(0)

            return image, label
        

# Copyright 2021 SVision Technologies LLC.
# https://github.com/AiviaCommunity/3D-RCAN/tree/master
def _get_spatial_ndim(x):
    return len(x.shape) - 2

def _conv(in_channels, out_channels, kernel_size, padding='same', **kwargs):
    n = _get_spatial_ndim(torch.zeros((1, in_channels, 1, 1, 1)))
    if n not in (2, 3):
        raise NotImplementedError(f'{n}D convolution is not supported')

    Conv = nn.Conv2d if n == 2 else nn.Conv3d
    return Conv(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

def _global_average_pooling(x):
    n = _get_spatial_ndim(x)
    if n == 2:
        return F.adaptive_avg_pool2d(x, 1)
    elif n == 3:
        return F.adaptive_avg_pool3d(x, 1)
    else:
        raise NotImplementedError(f'{n}D global average pooling is not supported')

class ChannelAttentionBlock(nn.Module):
    def __init__(self, num_channels, reduction):
        super().__init__()
        self.global_avg_pool = lambda x: _global_average_pooling(x)
        self.conv1 = _conv(num_channels, num_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv(num_channels // reduction, num_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, num_channels, channel_reduction=8, residual_scaling=1.0):
        super().__init__()
        self.conv1 = _conv(num_channels, num_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv(num_channels, num_channels, 3, padding=1)
        self.ca = ChannelAttentionBlock(num_channels, channel_reduction)
        self.residual_scaling = residual_scaling

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.ca(x)
        if self.residual_scaling != 1.0:
            x = x * self.residual_scaling
        x += res
        return x

class RCAN(nn.Module):
    def __init__(self, input_shape=(16, 256, 256, 1), num_channels=32, num_residual_blocks=5,
                 num_residual_groups=5, channel_reduction=8, residual_scaling=1.0, num_output_channels=-1):
        super().__init__()
        if num_output_channels < 0:
            num_output_channels = input_shape[1]

        self.standardize = lambda x: 2 * x - 1
        self.destandardize = lambda x: 0.5 * x + 0.5

        self.input_conv = _conv(input_shape[1], num_channels, 3, padding=1)
        
        self.residual_groups = nn.ModuleList([
            nn.Sequential(*[
                ResidualChannelAttentionBlock(num_channels, channel_reduction, residual_scaling)
                for _ in range(num_residual_blocks)
            ])
            for _ in range(num_residual_groups)
        ])
        
        self.conv_after_groups = _conv(num_channels, num_channels, 3, padding=1)
        self.output_conv = _conv(num_channels, 1, 3, padding=1)

    def forward(self, x):
        x = self.standardize(x)
        x = self.input_conv(x)
        
        long_skip = x
        
        for group in self.residual_groups:
            short_skip = x
            x = group(x)
            if len(self.residual_groups) > 1:
                x = self.conv_after_groups(x)
                x += short_skip
        
        x = self.conv_after_groups(x)
        x += long_skip
        
        x = self.output_conv(x)
        x = self.destandardize(x)
        
        return x

def build_rcan(input_shape=(16, 256, 256, 1), **kwargs):
    return RCAN(input_shape, **kwargs)
    


class RegressionPLModel(pl.LightningModule):
    def __init__(self, pred_shape, size=256, with_norm=False, total_steps=780):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func3 = nn.MSELoss()
        self.loss_func = lambda x,y: self.loss_func3(x, y) #0.3*self.loss_func1(x, y) + 0.5*self.loss_func2(x, y) + 0.2*self.loss_func3(x, y)

        self.rcan = build_rcan(input_shape=(1, CFG.in_chans, CFG.size, CFG.size), num_channels=32, num_residual_blocks=5, num_residual_groups=5)
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

    def forward(self, x):

        if self.hparams.with_norm:
            x = self.normalization(x)

        x = torch.reshape(x, (-1, CFG.in_chans, 1, CFG.size, CFG.size))
        pred_mask = self.rcan(x)
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")

        self.log("train/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x, y, xyxys = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"])

        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=CFG.lr)
        scheduler = get_scheduler(CFG, optimizer)

        return [optimizer], [scheduler]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
   


cfg_init(CFG)
wandb.init(project=CFG.comp_name)

valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"segments/{CFG.valid_id}/{CFG.valid_id}_T{CFG.exp_no}_inklabels.png", 0)
pred_shape = valid_mask_gt.shape

# Read images, labels and masks
train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(CFG.fragments)
valid_xyxys = np.stack(valid_xyxys)

# Create datasets
train_dataset = CustomDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))

valid_dataset = CustomDataset(
    valid_images, CFG, xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

# Create dataloaders
train_loader = DataLoader(train_dataset,
                            batch_size=CFG.train_batch_size,
                            shuffle=True,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                            )

valid_loader = DataLoader(valid_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

# Create the logger
wandb_logger = WandbLogger(project=CFG.comp_name, name=f'{CFG.backbone}_S2_T{CFG.trial_no}_E{CFG.exp_no}')
if CFG.ckpt:
  print(f"Loading checkpoint: {CFG.ckpt}")
  model = RegressionPLModel.load_from_checkpoint(CFG.ckpt, strict=False)
else:
  model = RegressionPLModel(pred_shape=pred_shape, size=CFG.size, total_steps=len(train_loader))


wandb_logger.watch(model, log="all", log_freq=CFG.print_freq)
trainer = pl.Trainer(
    max_epochs=CFG.epochs,
    accelerator="gpu",
    devices=1,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    default_root_dir="./models",
    accumulate_grad_batches=1,
    precision='16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    strategy='ddp_find_unused_parameters_true',
    callbacks=[ModelCheckpoint(filename=f'resnet3d-101-pretrained_T{CFG.trial_no}_E{CFG.exp_no}_size={CFG.size}_{CFG.valid_id}_'+'{epoch}', dirpath=CFG.model_dir, monitor='train/total_loss', mode='min', save_top_k=3)]
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

wandb.finish()