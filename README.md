# Embeddeding Multiple Deep Neural Networks for Adaptive Inference

This work is an extension of [**Learning Shared Filter Bases for Efficient ConvNets**](https://github.com/ssregibility/Net_RL2). 
- In our previous work, we explored the parameters sharing among convolution layers in a single DCNN. 
- In this work, we explore the parameter sharing among mutltiple networks embedded in a single model. 
- Networks with different computing costs and accuracy can be chosen at runtime according to changing demands. 

## Requirements

We conducted experiments under
- python 3.6.9
- pytorch 1.5, torchvision 0.4, cuda10

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Training

To train the models in the paper on CIFAR-10, run this command:

```train
python3 train_cifar10.py --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --lambdaR=10 --shared_rank=16 --unique_rank=1 --batch_size=256 --model=ResNet56_DoubleShared
```

## Evaluation

To evaluate proposed models on CIFAR-10, run:

```eval
python3 eval_cifar10.py --pretrained=<path_to_model> --model=ResNet56_DoubleShared --shared_rank=16 --unique_rank=1
```

## Pretrained Models

### CIFAR-10 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |   |
| ------------------ |---------------- | ------------ | ----- | ----- |
| ResNet32-S8U1      |     8.08%         |      0.15M     |  0.10G  | [Download](https://drive.google.com/file/d/1QmKmICZKk6h_FnctIr6LQrtFCCvWtcac/view?usp=sharing) |
| ResNet32-S16U1     |     7.43%         |      0.20M     |  0.16G  | [Download](https://drive.google.com/file/d/1cpCYf6iwN27RIDjmPxPSTXUW3htZ8-P5/view?usp=sharing) |
| ResNet56-S8U1      |     7.52%         |      0.20M     |  0.17G  | [Download](https://drive.google.com/file/d/1wUB3PnZ8lnSqXFTWGEk1eoLseSFQ2-Tj/view?usp=sharing) |
| ResNet56-S16U1     |     7.46%         |      0.22M     |  0.30G | [Download](https://drive.google.com/file/d/17rwH4_KNGX2nBgF0PBbBeKfve5IudZrY/view?usp=sharing) |
| ResNet32-S16U1\*    |     6.93%         |      0.24M     |  0.30G  | [Download](https://drive.google.com/file/d/1ZB5yZgMUhU9TGruZpInwX9UQo8kZXEHH/view?usp=sharing) |
| ResNet56-S16U1\*    |     6.30%         |      0.31M     |  0.30G  | [Download](https://drive.google.com/file/d/1zBQTvDYdbqnfdX3NA6mYy0lHvn68ANRl/view?usp=sharing) |

