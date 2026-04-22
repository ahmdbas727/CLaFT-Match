# CLaFT-Match: Closed-Loop Adaptive Feedback Thresholding for Class-Imbalanced Semi-Supervised Learning

This repository contains the implementation and experimental configuration for **CLaFT-Match**, a class-imbalanced semi-supervised learning framework that improves pseudo-label selection through **closed-loop adaptive thresholding** and **robust distribution correction**.

CLaFT-Match addresses two central challenges in class-imbalanced semi-supervised learning: **distribution bias** and **selection bias**. Instead of relying on a fixed confidence threshold for all classes, the method dynamically adjusts pseudo-label acceptance in a closed-loop manner according to class-wise learning status. At the same time, it applies robust adaptive distribution correction to calibrate pseudo-label predictions and better align class utilization during training.

## Overview

CLaFT-Match is designed for **class-imbalanced semi-supervised image classification**, especially under long-tailed labeled and unlabeled distributions where majority classes are over-represented and minority classes are easily under-trained. The framework contains the following key components:

- weak-view prediction for unlabeled samples
- Robust Adaptive Distribution Correction (R-ADC)
- Closed-Loop Class-Adaptive Thresholding (CL-CAT)
- class-aware pseudo-label selection
- consistency regularization with calibrated pseudo-labels

By coupling class-distribution correction with adaptive pseudo-label filtering, CLaFT-Match improves the balance, reliability, and effectiveness of unlabeled sample utilization during training.

## Method Summary

For each unlabeled sample, CLaFT-Match first computes a weak-view prediction and confidence score. It then estimates the class distribution of unlabeled data using a robust adaptive correction mechanism and calibrates the predicted probabilities accordingly. Based on the current class-wise training feedback, the method dynamically updates class-specific thresholds to determine whether pseudo-labels should be accepted. The selected pseudo-labels are then used to supervise the unsupervised consistency objective on strongly augmented views.

Instead of applying a single global threshold to all classes, CLaFT-Match forms a closed-loop interaction between prediction calibration and threshold adaptation, which helps reduce over-selection of majority classes and under-selection of minority classes.

## Datasets

CLaFT-Match is evaluated on the following benchmark datasets:

- **CIFAR-10-LT**
- **SVHN**
- **STL-10**

The method is tested under different class-imbalance settings, including **matched** and **mismatched** labeled/unlabeled distributions, as well as severe long-tailed scenarios with limited labeled data.

## Environment

We recommend creating the environment from the provided configuration file:

```bash
conda env create -f environment.yml
conda activate ssl




## Main Hyperparameters

The main hyperparameter settings used in CLaFT-Match include:
backbone: WideResNet-28-2
optimizer: SGD with Nesterov momentum
learning rate: 0.03
weight decay: 5e-4
batch size: 64
unlabeled ratio: μ = 2
total epochs: 512
EMA teacher decay: 0.9995
unlabeled distribution momentum: 0.999
base confidence threshold: adaptive by class
unsupervised loss weight: λu = 1
For different datasets and imbalance settings, the implementation uses adjusted configurations consistent with the experimental protocol.






## Main Results

Representative results of CLaFT-Match are summarized below:
Dataset	Setting	Result
CIFAR-10-LT	N1 = 1500, M1 = 3000, γ = 50	91.95 accuracy
CIFAR-10-LT	N1 = 1500, M1 = 3000, γ = 100	89.26 accuracy
CIFAR-10-LT	N1 = 1500, M1 = 3000, γ = 150	87.87 accuracy
CIFAR-10-LT	N1 = 500, M1 = 4000, γ = 50	87.43 accuracy
CIFAR-10-LT	N1 = 500, M1 = 4000, γ = 100	83.88 accuracy
CIFAR-10-LT	N1 = 500, M1 = 4000, γ = 150	76.93 accuracy
CIFAR-10-LT	mismatched, γu = 1	92.75 accuracy
CIFAR-10-LT	mismatched, γu = 1/100	81.32 accuracy
SVHN	γ = 100	96.21 accuracy
STL-10	γ = 10	86.12 accuracy
STL-10	γ = 20	81.18 accuracy
These results show that CLaFT-Match performs strongly across class-imbalanced semi-supervised learning benchmarks and maintains robust performance under both matched and mismatched distribution settings.




## Evaluation Metrics
The experimental evaluation includes multiple metrics:
Accuracy
Balanced Accuracy
Precision
Recall
F1-Score
These metrics provide a more complete assessment of model performance, especially under class imbalance where standard accuracy alone may not fully reflect minority-class behavior.





## Efficiency Notes
CLaFT-Match does not require explicit neighbor retrieval, graph construction, or memory-bank maintenance. Its main computation comes from weak/strong augmentation training, class-distribution estimation, probability calibration, and class-adaptive threshold updating. From a memory perspective, the method mainly maintains batch predictions, EMA statistics, and class-wise threshold states, making it lightweight and easy to integrate into standard FixMatch-style training pipelines.





## Notes
This repository contains the implementation of CLaFT-Match and the corresponding experimental setup. Before running the code, please make sure that dataset paths, training configuration, and hardware-related settings are adjusted to your local environment.



## License
Please follow the license terms of the source code and included components in this repository.