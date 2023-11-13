import os
import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
# from albumentations.pytorch.transforms import ToTensorV2
# import albumentations.augmentations.transforms as transforms
from timm.data import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pydicom as dicom


def build_loader_SIIM(config, ddp):
    image_root = config.DATA.IMAGE
    mask_root = config.DATA.MASK
    traintxt = config.DATA.TRAIN_LIST
    valtxt = config.DATA.VAL_LIST
    testtxt = config.DATA.TEST_LIST
    

    train_list = []
    train_mask = []
    val_list = []
    val_mask = []
    test_list = []
    test_mask = []

    if config.MODE == 'train':
        with open(traintxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                train_list.append(i.split('\n')[0])
        train_num = len(train_list)
        if config.RATIO == '50':
            train_list = train_list[:int(train_num/2)]
        elif config.RATIO == '25':
            train_list = train_list[:int(train_num/4)]
        elif config.RATIO == '1shot':
            train_list = [train_list[0]]
        elif config.RATIO == '5shot':
            train_list = train_list[:5]
        elif config.RATIO == '10shot':
            train_list = train_list[:10]
                
        with open(valtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                val_list.append(i.split('\n')[0])


        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = SIIM_dataset(image_root=image_root, mask_root=mask_root, datalist=train_list, img_transforms=img_train_transforms, config=config)
        
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = SIIM_dataset(image_root=image_root, mask_root=mask_root, datalist=val_list, img_transforms=img_val_transforms, config=config)
        

        if ddp:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset)

            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    # shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True,
                                    sampler=sampler_train)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                # sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS,
                                sampler=sampler_val)
        else:
            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                # sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)


        # setup mixup / cutmix
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                        mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                        prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                        label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
            
        return train_dataset, val_dataset, train_loader, val_loader, mixup_fn
    elif config.MODE == 'test':
        with open(testtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                test_list.append(i.split('\n')[0])
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = SIIM_dataset(image_root=image_root, mask_root=mask_root, datalist=test_list, img_transforms=img_test_transforms, config=config)
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class SIIM_dataset(Dataset):
    def __init__(self, image_root, mask_root, datalist, img_transforms, config):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.image_root = image_root
        self.mask_root = mask_root 
        self.datalist = datalist
        self.config = config

    def __getitem__(self, index):

        ds = dicom.dcmread(os.path.join(self.image_root,self.datalist[index]))
        image = np.array(ds.pixel_array)
        image = (np.array(image)).astype('uint8')
        if len(image.shape) == 2:
            image = np.repeat(image[..., None], 3, 2)

        mask = cv2.imread(os.path.join(self.mask_root, self.datalist[index]+'_mask.png'))
        mask[mask==255]=1

        transformed = self.img_transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'] 
        mask = mask.permute([2,0,1])
        mask = mask[0].unsqueeze(0)

        return image.float(), mask.float()
    # return image 
    def __len__(self):
        return len(self.datalist)

def img_transforms(mode, config):
    if mode == 'train':
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                            albumentations.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            albumentations.ShiftScaleRotate(rotate_limit=10),
                                            albumentations.RandomBrightnessContrast(),
                                            ToTensorV2()
                                            ])
        
    elif mode == 'val' or mode == 'test':
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                            albumentations.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            ToTensorV2()
                                            ])


    return data_transforms

