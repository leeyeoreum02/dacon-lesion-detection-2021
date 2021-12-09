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
- Ensemble 9 models above using wbf can achieve **0.7675 Public LB/0.8261 Prviate LB (4th/250)**

### Weights
- [Download weights zip file (.ckpt)](https://drive.google.com/file/d/1XmdUqAcj06DN4Hd9QLII0GdweV24bHjw/view?usp=sharing)

### Environment
- OS: Windows 11 Education Insider Preview
- [pip 가상환경을 사용할 경우](https://github.com/leeyeoreum02/dacon-lesion-detection-2021/blob/master/requirements.txt)
- [conda 가상환경을 사용할 경우](https://github.com/leeyeoreum02/dacon-lesion-detection-2021/blob/master/environment.yaml)

### Total Process
1. 대회 데이터를 다운로드하고 zip파일을 ./ori_data에 압축해제
2. ./ori_data/test에 있는 json 파일 각각의 imageData(base64 형식의 이미지 데이터)를 jpg 파일로 변환 ㅎㅜ './data/testㅇㅔ ㅈㅓㅈㅏㅇ
3. 각 모델의 각 fold별 weight로 추론 후 추론 결과를 coco format을 따르는 json 파일로 저장
4. 저장된 json 파일을 바탕으로 wbf 적용 후 결과를 json 파일로 저장
5. wbf가 적용된 json 파일을 csv 파일로 변환

## Quick Start
1. 
2.
3.
```
conda env create -f environment.yml
conda activate lesion
python eval.py
```