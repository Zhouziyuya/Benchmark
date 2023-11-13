# Benchmark
Benchmark training codes for classification and segmentation. The code can be used to train benchmarks on three backbone: swin-base, vit-base and resnet50 and the backbones can be modified as you like. The segementation architecture is UperNet.

## Training 

* For classification task, ddp is added in this code to accelerate the training processing via multiple gpus.
You can use several gpus and set `--nproc_per_node=gpu number`:
```
CUDA_VISIBLE_DEVICES="5,6" python -m torch.distributed.launch --nproc_per_node 2 --master_port=25641 poparval_main_ddp.py --img_size 448 --fold 1 --dataset NIHchest
```
Also, you can use one gpu:
```
CUDA_VISIBLE_DEVICES="5" python -m torch.distributed.launch --nproc_per_node 1 --master_port=25641 poparval_main_ddp.py --img_size 448 --fold 1 --dataset NIHchest
```

* For segmentation task, ddp is not added. `--local_rank` is used to set device number
```
python main_seg.py --backbone vit_base --pretrain_mode vit_seg_selfpatch --pretrain_weight /sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/SelfPatch_vit-b32_448/checkpoint0300.pth --local_rank 6 --dataset SIIM
```

## Testing

* For classification testing, TTA (TenCrop) is used to boost the performance. Use `--resume` to set your testing checkpoint.
```
python test.py --dataset NIHchest --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_adodocar_448_1/best.pth' --device 1
```

* For segmentation testing:
```
python test_seg.py --resume
```

## Some configrations

* Dataloaders
I write a dataloader file for each dataset which can be found in `.\data`. And I trained four classification datasets: CheXpert, NIHChestXray14, ShenzhenCXR and RSNA Pneumonia, segmentation datasets: JSRT, SIIM, ChestXdet, Montgomery


* Config

For each dataset, there is a config file locates in `.\configs`. You can modify the initial configrations in these files.

* Load pretrained model

The pretrained model loading function is in `utils.py` file. You may need to change the model keys' name for some pretrained model. Please set `load_checkpoint()` in `utils.py` file.