# MLDL 2022

Project: Domain adaptation in real-time semantic segmentation

Backbone model: BiseNet

## Model complexity

It is possible to print MACs, FLOPs and number of parameters of a model in this repository by using this script:
```
!pip install thop
!python complexity.py
```
An example with some parameters:
```
!python complexity.py --model "bisenet" --input_channels 3 --context_path "resnet18"
```
