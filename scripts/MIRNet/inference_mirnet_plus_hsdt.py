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
import scipy.stats as st

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
from tap import Tap
from tqdm.auto import tqdm

from  MIRNet_model import MIRNet
from hsdt import hsdt

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.login(key='574c23ca47bbd885a816336238f90cdee9d3a9ca')
wandb.init(project='Inference')

print("Running inference...")
start_idxs =     {"S4_20231210132040":[17],  "S3_20240618142020":[17],  "S3_20240716140050":[17],  "S3_20240712074250":[17],  "S1_20231205141500":[17],  "S1_20230929220920":[17],  "S1_20240110113230":[18],  "S1_20231004222109":[10], 'S1_20230515162442':[17],  '20230422011040':[17],  '20230422213203':[17],  '20230426114804':[17],  '20230522182853':[17],  '20230709155141':[17],  '20230801194757':[17],  '20240516205750':[1, 17, 35],  '20230517214715':[17],  '20230509173534':[17]}
reversed_load =  {"S4_20231210132040":False, "S3_20240618142020":False, "S3_20240716140050":False, "S3_20240712074250":False, "S1_20231205141500":False, "S1_20230929220920":False, "S1_20240110113230":False, "S1_20231004222109":False,'S1_20230515162442':False, '20230422011040':False, '20230422213203':False, '20230426114804':False, '20230522182853':False, '20230709155141':False, '20230801194757':False, '20240516205750':False, '20230517214715':False, '20230509173534':False}
#reversed_load = {"S4_20231210132040":True,  "S3_20240618142020":True,  "S3_20240716140050":True,  "S3_20240712074250":True,  "S1_20231205141500":True,  "S1_20230929220920":True,  "S1_20240110113230":True,  "S1_20231004222109":True, 'S1_20230515162442':True,  '20230422011040':True,  '20230422213203':True,  '20230426114804':True,  '20230522182853':True,  '20230709155141':True,  '20230801194757':True,  '20240516205750':True,  '20230517214715':True,  '20230509173534':True}
flipped =        {"S4_20231210132040":False, "S3_20240618142020":False, "S3_20240716140050":False, "S3_20240712074250":False, "S1_20231205141500":False, "S1_20230929220920":False, "S1_20240110113230":False, "S1_20231004222109":False,'S1_20230515162442':False, '20230422011040':False, '20230422213203':False, '20230426114804':False, '20230522182853':False, '20230709155141':False, '20230801194757':False, '20240516205750':False, '20230517214715':False, '20230509173534':False}
#flipped =       {"S4_20231210132040":True,  "S3_20240618142020":True,  "S3_20240716140050":True,  "S3_20240712074250":True,  "S1_20231205141500":True,  "S1_20230929220920":True,  "S1_20240110113230":True,  "S1_20231004222109":True, 'S1_20230515162442':True,  '20230422011040':True,  '20230422213203':True,  '20230426114804':True,  '20230522182853':True,  '20230709155141':True,  '20230801194757':True,  '20240516205750':True,  '20230517214715':True,  '20230509173534':True}
trial_no = 18
exp_no = 8
epoch = 5
val_id = 'S1_20230515162442'
model_type =  "ft"
TRAIN_ON_DIFF = False
TRAIN_SOBEL = False
class InferenceArgumentParser(Tap):
    segment_id:list[str] = ['20240516205750'] #['S4_20231210132040', 'S3_20240716140050', 'S3_20240618142020', '20240516205750', 'S1_20231004222109'] #["S1_20231004222109", '20240516205750', "S3_20240618142020", "S4_20231210132040"]#["S1_20230929220920", "S3_20240618142020", '20240516205750', "S4_20231210132040", "S1_20231004222109", '20230422213203', '20230801194757', '20230522182853', '20230509173534', '20230709155141']
    segment_path:str='./'
    model_path:str = "/content/drive/MyDrive/Scrolls/ckpts/Vesuvius_S2_T18/Training_S2_T18_E8/Vesuvius_S2_T18-models/MIRNet-Plus-HSDT_T18_E8_size=64_S1_20230515162442_epoch=5.ckpt"
    out_path:str = "/content/"
    in_chans:int = 20
    stride: int = 2
    workers: int = 2
    batch_size: int = 256
    size:int = 64
    reverse:int = 0
    device:str = 'cuda'
    format = 'tif'


args = InferenceArgumentParser().parse_args()




class CFG:
    # ============== comp exp name =============
    comp_name = 'Inference'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    exp_name = 'Inference'

    # ============== model cfg =============
    in_chans = args.in_chans

    # ============== training cfg =============
    size = args.size
    tile_size = size
    stride = tile_size // 3

    train_batch_size = args.batch_size
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 30

    # adamW warmup あり
    warmup_factor = 10
    lr = 1e-4 / warmup_factor
    min_lr = 1e-6
    num_workers = 1
    seed = 42

    # ============== augmentation =============
    CLAHE = A.Compose([A.CLAHE(clip_limit=(1, 16), tile_grid_size=(4, 4), p=1)])
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]



def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()



def set_seed(seed=42, cudnn_deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False



def cfg_init(cfg, mode='val'):
    set_seed(cfg.seed)



def aggregate_layers(fragment_id, i, n, max_layer_number=65):
    start = i-n
    end = i+n+1

    assert start > -1, "Negative layer number."
    assert end <= max_layer_number, "Exceeded max layer number."

    img = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(start):02}.tif", 0).astype(np.float32)
    for j in range(start+1, end):
        img += cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(j):02}.tif", 0).astype(np.float32)
            
    img = img/(2*n+1)
    img = img.astype(np.uint8)

    return img


def read_image_mask(fragment_id, start_idx=17, end_idx=43, rotation=0):
    images = []
    print(fragment_id)
    print(start_idx, end_idx, rotation)
    if reversed_load[fragment_id]:
      print("Reversed Load")
      idxs = range(64, 64-CFG.in_chans, -1)
    else:
      idxs = range(start_idx, end_idx)

    print(list(idxs))

    dims = None
    if TRAIN_ON_DIFF:
      print("TRAINED ON DIFFs")
      for i in idxs:
            image0 = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{i:02}.tif", 0).astype(np.float32)
            #image0 = cv2.GaussianBlur(image0,(7,7),0).astype(np.float32)

            image1 = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+1):02}.tif", 0).astype(np.float32)
            #image1 = cv2.GaussianBlur(image1,(7,7),0).astype(np.float32)

            image2 = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+2):02}.tif", 0).astype(np.float32)
            #image2 = cv2.GaussianBlur(image2,(7,7),0).astype(np.float32)

            image3 = cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{(i+3):02}.tif", 0).astype(np.float32)
            #image3 = cv2.GaussianBlur(image3,(7,7),0).astype(np.float32)

            if TRAIN_SOBEL:
              print("TRAINING ON SOBEL DIFFs")
              gX = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
              gY = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
              image1 = (gX**2 + gY**2)**0.5

              gX = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
              gY = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
              image2 = (gX**2 + gY**2)**0.5

            d1 = abs(image1 - image0)
            d2 = abs(image2 - image1)
            d3 = abs(image3 - image2)
            image = (d1 + d2 + d3)/3
            image = image.astype(np.uint8)
            #image = CFG.CLAHE(image=image)['image']

            pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
            pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
            
            dims = image.shape
            
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            images.append(image)

    else:
      for i in idxs:
          image = aggregate_layers(fragment_id, i, 1)#cv2.imread(CFG.comp_dataset_path + f"segments/{fragment_id}/layers/{i:02}.tif", 0)#
          if rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          elif rotation == -90 or rotation == 270 :
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
          elif rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)

          dims = image.shape
          pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
          pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
          image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
          #image = np.clip(image, 0, 200)
          images.append(image)

    images = np.stack(images, axis=2)
    if flipped[fragment_id]:
        print("Reverse Segment")
        images = images[:,:,::-1]

    fragment_mask = np.ones_like(images[:, :, 0]) * 255
    print(images.shape, fragment_mask.shape)
    return images, fragment_mask, dims



def get_img_splits(fragment_id, s, e, rotation=0):
    images = []
    xyxys = []
    image, fragment_mask, dims = read_image_mask(fragment_id, s, e, rotation)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])

    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    
    return test_loader, np.stack(xyxys), (image.shape[0],image.shape[1]), fragment_mask, dims



def get_transforms(data, cfg):
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug



class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].squeeze(0)

        return image, xy



class RegressionPLModel(pl.LightningModule):
    def __init__(self, pred_shape, size=256, with_norm=False, total_steps=780):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        #self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        #self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func3 = nn.SmoothL1Loss(beta=0.5)
        self.loss_func = lambda x, y: self.loss_func3(x, y)
        #self.loss_func = lambda x,y : 0.5*self.loss_func1(x, y) + 0.5*self.loss_func2(x, y)
        self.inp = nn.Conv2d(CFG.in_chans, 31, 3, padding='same')
        self.mirnet = MIRNet(in_channels=31, out_channels=31)
        self.hsdt = hsdt()
        self.out = nn.Conv3d(31, 1, 3, padding='same')

        #self.out = nn.Conv2d(CFG.in_chans, 1, 3, padding='same')
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)


    def forward(self, x):

        if self.hparams.with_norm:
            x = self.normalization(x)
        
        #print(x.shape)
        x = self.inp(x)
        #print(x.shape)
        x = self.mirnet(x)
        #print(x.shape)
        x = x.unsqueeze(1)
        #print(x.shape)
        x = self.hsdt(x)
        #print(x.shape)
        x = torch.permute(x, (0, 2, 1, 3, 4))
        #print(x.shape)
        x = self.out(x)
        #print(x.shape)
        
        return x
    

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
        self.log.log_image(key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"])

        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.lr)
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
   


def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel = gkern(CFG.size, 1)
    kernel = kernel/kernel.max()

    for _, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)

        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(y_preds[i].squeeze(0).squeeze(0).numpy(), kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    return mask_pred



if __name__ == "__main__":
    print(f"Model_Path:{args.model_path}")
    cfg_init(CFG)
    model = RegressionPLModel.load_from_checkpoint(args.model_path, strict=False)
    model.cuda()
    model.eval()
    
    for fragment_id in args.segment_id:
        if os.path.exists(f"{args.segment_path}/segments/{fragment_id}/layers/17.{args.format}"):
            for r in [0]:
                for i in start_idxs[fragment_id]:
                    start_f = i
                    end_f = start_f + args.in_chans
                    test_loader, test_xyxz, test_shape, fragment_mask, dims = get_img_splits(fragment_id, start_f, end_f, r)
                    height, width = dims
                    mask_pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
                    mask_pred = mask_pred[:height, :width]
                    mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
                    mask_pred /= mask_pred.max()

                    img = wandb.Image(
                    mask_pred, 
                    caption=f"{fragment_id}_T{trial_no}_E{exp_no}_Rot={r}_{start_f}-{end_f}_val={val_id}_{model_type}_epoch={epoch}_R={flipped[fragment_id]}_TOD={TRAIN_ON_DIFF}_predictions"
                    )
                    wandb.log({f'{fragment_id}_T{trial_no}_E{exp_no}_Rot={r}_{start_f}-{end_f}_val={val_id}_{model_type}_epoch={epoch}_R={flipped[fragment_id]}_TOD={TRAIN_ON_DIFF}_predictions':img})
                    if len(args.out_path) > 0:
                        # CV2 image
                        image_cv = (mask_pred * 255).astype(np.uint8)
                        try:
                            os.makedirs(args.out_path, exist_ok=True)
                        except:
                            pass
                        cv2.imwrite(os.path.join(args.out_path, f"{fragment_id}_T{trial_no}_E{exp_no}_Rot={r}_{start_f}-{end_f}_val={val_id}_{model_type}_epoch={epoch}_R={flipped[fragment_id]}_prediction.png"), image_cv)
                    
                    gc.collect()
                    del mask_pred, test_loader, test_xyxz, test_shape, fragment_mask, img
        else:
          print("Check the entry condition for inference")


    del model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()