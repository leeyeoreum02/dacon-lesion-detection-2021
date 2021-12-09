import os
import json
import torch
import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


torch.multiprocessing.set_sharing_strategy('file_system')


class EfficientDetDataset(Dataset):
    def __init__(self, fold_id, image_dir, annot_path, transforms=None, **kwargs):
        with open(annot_path, 'r') as f:
            annotations = json.load(f)

        self.fold_id = fold_id
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = {annot['id']: {
            'file_name': annot['file_name'], 
            'height': annot['height'],
            'width': annot['width']
        } for annot in annotations['images']}

        self.annotations = defaultdict(list)
        for annot in annotations['annotations']:
            self.annotations[annot['image_id']].append({
                'category_id': annot['category_id'],
                'bbox': np.array(annot['bbox']),
                'area': annot['area']
            })

    def __len__(self):
        if self.fold_id:
            return len(self.fold_id)
        else:
            return len(self.image_ids)
    
    def __getitem__(self, index):
        if self.fold_id:
            index = self.fold_id[index]
            
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, image_id['file_name'])).convert('RGB')
        width, height = image_id['width'], image_id['height']

        boxes = []
        labels = []
        for annot in self.annotations[index]:
            x1, y1, w, h = annot['bbox']
            x2, y2 = x1 + w, y1 + h
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, width)
            y2 = min(y2, height)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(annot['category_id'])

        boxes = np.asarray(boxes)
        labels = np.asarray(labels)

        target = {
            'image': np.asarray(image, dtype=np.uint8),
            'bboxes': boxes,
            'labels': labels
        }

        if self.transforms is not None:
            target = self.transforms(**target)

        image = target['image']
        del target['image']
   
        target['bboxes'] = np.asarray(target['bboxes'])
        target['bboxes'][..., [0, 1, 2, 3]] = target['bboxes'][..., [1, 0, 3, 2]]
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target.update({
            'image_id': torch.as_tensor([index]),
            'img_scale': torch.as_tensor([1.0]),
            'img_size': (width, height)
        })
        
        return image, target, index
    
    
class EfficientDetEvalDataset(Dataset):
    def __init__(self, image_dir, transforms=None, **kwargs):
        self.transforms = transforms
        self.image_ids = sorted(glob(os.path.join(image_dir, '*.jpg')))

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(image_id).convert('RGB')
        width, height = image.size
        image = np.asarray(image, dtype=np.uint8)
        
        target = {
            'image_size': (width, height),
        }

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, target, image_id


class EfficientDetDataModule(LightningDataModule):
    def __init__(
        self,
        image_dir='./data/',
        train_fold_id=None,
        valid_fold_id=None,
        train_transforms=None,
        valid_transforms=None,
        predict_transforms=None,
        num_workers=4,
        batch_size=8
    ):
        super().__init__()
        self.train_fold_id = train_fold_id
        self.valid_fold_id = valid_fold_id
        self.image_dir = image_dir
        self.train_image_dir = os.path.join(self.image_dir, 'train')
        self.annot_path = os.path.join(self.train_image_dir, 'annotations.json')
        self.predict_image_dir = os.path.join(self.image_dir, 'test')
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.predict_transforms = predict_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EfficientDetDataset(
            self.train_image_dir, self.train_fold_id, self.annot_path, self.train_transforms)
        self.valid_dataset = EfficientDetDataset(
            self.train_image_dir, self.valid_fold_id, self.annot_path, self.valid_transforms)
        self.predict_dataset = EfficientDetEvalDataset(
            self.predict_image_dir, self.predict_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.predict_collate_fn
        )

    @staticmethod
    def train_collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images).float()

        boxes = [torch.from_numpy(target['bboxes']).type(torch.float32) for target in targets]
        labels = [target['labels'].float() for target in targets]
        img_sizes = torch.tensor([target['img_size'] for target in targets]).float()
        img_scales = torch.tensor([target['img_scale'] for target in targets]).float()

        annotations = {
            'bbox': boxes,
            'cls': labels,
            'img_size': img_sizes,
            'img_scale': img_scales,
        }

        del targets

        return images, annotations
    
    @staticmethod
    def predict_collate_fn(batch):
        images, targets, image_id = tuple(zip(*batch))
        images = torch.stack(images).float()
        image_sizes = torch.tensor([target['image_size'] for target in targets]).float()

        annotations = {
            'image_size': image_sizes,
            # 'img_scale': img_scales,
        }

        del targets

        return images, annotations, image_id
    

if __name__ == '__main__':
    # dataset = EfficientDetDataset('./data/', './data/annotations.json', get_train_transforms())
    # print(dataset[1])
    data_loader = EfficientDetDataModule()
    print(data_loader.on_train_dataloader)