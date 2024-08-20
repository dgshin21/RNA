# RNA
This repository contains code and the real world dataset for our paper: [Representation Norm Amplification for
Out-of-Distribution Detection in Long-Tail Learning](https://openreview.net/forum?id=z4b4WfvooX).


## Dependencies
- Python 3.9.13
- Pytorch 1.12.1
- Torchvision 0.13.1
- Numpy 1.23.1
- CUDA 11.5 
- PIL 9.5.0
- scikit-learn 1.2.2

## How to get the dataset
### CIFAR 
You can get the CIFAR10 and CIFAR100 training and test dataset from the official website: https://www.cs.toronto.edu/~kriz/cifar.html.
- CIFAR10<br>
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
- CIFAR100<br>
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

### ImageNet 
You can get the ImageNet datasets from the official website after log-in: https://image-net.org/download-images.php
- ImageNet-1k (ILSVRC2012)

  Training set:
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```
  Test set: 
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```
- ImageNet-10k<br>
```bash
wget https://image-net.org/data/imagenet10k_eccv2010.tar
```
 
- ImageNet-Extra<br>
```bash
python datasets/extract_extra_imagenet_1k.py --drp ../data
```
- ImageNet-1k-OOD<br>
```bash
python datasets/extract_ood_test_imagenet_1k.py --drp ../data
```

## How to run the code
### Train
- CIFAR10-LT <br>
python train.py --ds cifar10 --drp <where_you_store_all_your_datasets> --suffix exp0 --gpu 0

- CIFAR100-LT <br>
python train.py --ds cifar100 --drp <where_you_store_all_your_datasets> --suffix exp0 --gpu 0

- ImageNet-LT <br>
python train.py --ds imagenet --drp <where_you_store_all_your_datasets> --lr 0.1 --epochs 100 --model ResNet50 --suffix exp0 --ddp --gpu 0,1,2,3,4,5,6,7 

You can experiment on various settings of "Lambda" or "imbalance_ratio" by running the following as an example:<br>
python train.py --ds cifar10 --suffix Lambda0.6 --Lambda 0.6 --gpu 0 <br>
python train.py --ds cifar100 --suffix rho0.1 --imbalance_ratio 0.1 --gpu 0

### Test
- CIFAR10-LT<br>
for dout in texture svhn cifar tin lsun places365 <br>
do<br>
python test.py --gpu 0 --ds cifar10 --dout $dout \\ <br>
    --drp <where_you_store_all_your_datasets> \\ <br>
    --ckpt_path <where_you_save_the_ckpt> <br>
done

- CIFAR100-LT<br>
for dout in texture svhn cifar tin lsun places365 <br>
do<br>
python test.py --gpu 0 --ds cifar100 --dout $dout \\ <br>
    --drp <where_you_store_all_your_datasets> \\ <br>
    --ckpt_path <where_you_save_the_ckpt> <br>
done
- ImageNet-LT<br>
python test.py --ds imagenet --model ResNet50 --batch_size 256 --ckpt_path \<where you save the ckpt>

The test code will print out the evaluation results of 3 OOD performance metrics (AUROC, AUPR, FPR95) 
and 4 classification performance metrics (ACC, Many, Medium, Few).

### Acknowledgement
Part of our codes are adapted from these repos:

pytorch-cifar - https://github.com/kuangliu/pytorch-cifar - MIT license

outlier-exposure - https://github.com/hendrycks/outlier-exposure - Apache-2.0 license

Long-Tailed-Recognition.pytorch - https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch - GPL-3.0 license

PASCL - https://github.com/amazon-science/long-tailed-ood-detection - Apache License 2.0
