# Image Classification with a Pretrained CNN Ensemble

Fine-tunes ImageNet-pretrained **ResNet-50** and **VGG-16** models and combines them in a soft-voting ensemble to classify 32×32 images into 10 classes. Built for Data Challenge 1 of **STAT 940 – Deep Learning** at the University of Waterloo ([Kaggle competition](https://www.kaggle.com/competitions/stat940f23dc1)).

## Results

Validation accuracy on a 10% hold-out set (5,000 images):

| Model | Validation Accuracy | Test Accuracy |
|---|---|---|
| ResNet-50 | 92.0% | - |
| VGG-16 | 94.5% | - |
| **Ensemble (Soft Voting)** | **95.1%** | **66.7%**| 

## Approach

- **Data:** 50,000 labelled training images and 10,000 test images (32×32 RGB), split 90/10 into training and validation sets.
- **Models:** ImageNet-pretrained ResNet-50 (`resnet50.a1_in1k`) and VGG-16 (`vgg16_bn.tv_in1k`), loaded from the Hugging Face Hub with `timm`. The 1000-class classifier is replaced with a 10-class one, and the whole network is fine-tuned.
- **Preprocessing:** images are upsampled to 224×224 and normalized with ImageNet statistics. Training images also get random horizontal flips and ±5° rotations.
- **Training:** AdamW, a cosine learning-rate schedule with warm-up, and mixed precision, for 5 epochs per model on an NVIDIA RTX 5090. The best epoch is kept.
- **Ensemble:** the two models' predicted class probabilities are averaged (soft voting).

## Author

Giovanni (Jonathan) Zingaro · [GitHub](https://github.com/jonathanzingaro-jpg) · [LinkedIn](https://www.linkedin.com/in/jonathanzingaro/)
