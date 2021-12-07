# Dacon 병변 검출 AI 경진대회

## Summary
- Heavy augmentation
- EfficientDet
- Ensemble multi-scale model: Weighted-Boxes-Fusion  

### Heavy augmentation
- CLAHE, RandomBrightnessContrast, ColorJitter, RGBShift, RandomSnow, RandomCrop, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90

### Model
- 5 folds
- Optimizer: Adam with initial LR 5e-4 for EfficientDet
- Mixed precision training with pytorch lightning

### Performance
Public LB AP/Private LB AP

- EfficientDet-d7 image-size 512: Fold0 0.725 Fold1 - Fold2 - Fold3 -
- EfficientDet-d5 image-size 384: Fold0 0.699 Fold1 0.709 Fold2 - Fold3 0.710 Fold4 0.697
- Ensemble 9 models above using wbf can achieve **0.7675 Public LB/0.8261 Prviate LB (4/250)**