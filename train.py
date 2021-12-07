# from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
import os
import json
import random
import argparse
import pandas as pd
from sklearn.model_selection import KFold
from pytorch_lightning.plugins import DDPPlugin
from lib.model_effdet import EfficientDetModel
from lib.dataset import EfficientDetDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def get_args():
    parser = argparse.ArgumentParser(description='Training EfficientDet')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--architecture_name', type=str, default='tf_efficientdet_d4')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    return args


def get_train_transforms(image_size):
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.RGBShift(p=0.2),
            A.RandomSnow(p=0.2),
            # A.RandomCrop(height=320, width=320, p=0.4),
            # A.ShiftScaleRotate(
            #     scale_limit=0.2, 
            #     rotate_limit=10, 
            #     p=0.4
            #  ),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(p=0.2),
            A.RandomRotate90(p=0.2),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels'],
        ),
    )


def get_valid_transforms(image_size):
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels']
        ),
    )
    
    
def split_dataset(annot_path: os.PathLike, save_path: os.PathLike, seed: int = 42) -> None:
    with open(annot_path, 'r') as f:
        annotations = json.load(f)
    
    image_infos = annotations['images']
    random.Random(seed).shuffle(image_infos)
    
    ids = [annot['id'] for annot in image_infos]
    file_names = [annot['file_name'] for annot in image_infos]
    
    df = pd.DataFrame({'id': ids, 'file_name': file_names})
    kfold = KFold(n_splits=5)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)
        
    df.to_csv(save_path, index=False)


def train(model_name, args, fold):
    df = pd.read_csv('./data/train/split_kfold.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    train_fold_id = list(df_train['id'])
    valid_fold_id = list(df_valid['id'])
    
    data_module = EfficientDetDataModule(
        train_fold_id=train_fold_id,
        valid_fold_id=valid_fold_id,
        image_dir='./data/',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        train_transforms=get_train_transforms(image_size=args.image_size),
        valid_transforms=get_valid_transforms(image_size=args.image_size),
    )

    model = EfficientDetModel(
        mode='train',
        model_architecture=args.architecture_name,
        num_classes=4,
        image_size=args.image_size,
        learning_rate=args.lr,
    )
    
    ckpt_path = f'./weights/{model_name}/{fold}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map50',
        dirpath=ckpt_path,
        filename='{epoch}-{val_map50:.2f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=[0, 1, 2, 3],
        # gpus=1,
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=16,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data_module)


def main():
    args = get_args()
    
    split_dataset('./data/train/annotations.json', './data/train/split_kfold.csv')
    
    for k in range(5):
        train('effdetd7i512', args, k)


if __name__ == '__main__':
    main()