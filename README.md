# Vision Transformers for Image Classification

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project 
Vision Transformers are the future for Computer Vision tasks. As stated in the papers, Vision Transformers give the state of the art performance on ImageNet dataset and outperform the most popular CNN architectures like Resnet-50 etc. However, they take up a lot of memory and using a hybrid approach where the CNN architecture gives the feature map and then feeding that feature map as an input to the Vision Transformer model requires less model size as compared to the other ones. This repository has the ViT_Base and ViT_Hybrid(using Resnet-50 as backbone) and tries to compare their performance on the Dog/Cat dataset.


### Data
The dataset used was Dog/Cat dataset for the task of image classification(number of classes=2).


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/lukemelas/PyTorch-Pretrained-ViT