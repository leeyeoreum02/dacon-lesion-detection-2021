import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image
from glob import glob

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion

from lib.model_effdet import EfficientDetModel
from lib.dataset import EfficientDetDataModule
from lib.visualize import draw_bbox_result
from lib.utils import convert_to_jpg


def get_args():
    parser = argparse.ArgumentParser(description='Evaluating EfficientDet')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--architecture_name', type=str, default='tf_efficientdet_d4')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    return args


def get_predict_transforms(image_size):
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )
    
    
def get_result(outputs, save_dir, fold, epoch, model_name='effdetd5'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    res_json = []
    for batch in tqdm(outputs):
        for image_id, boxes, labels, scores in zip(*batch):
            image_id = int(image_id.split('\\')[-1].split('.')[0].split('_')[-1])
            boxes = np.array(boxes, dtype=np.float32)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            boxes[:, 2] = widths
            boxes[:, 3] = heights
            labels = np.array(labels, dtype=np.uint8)
            scores = np.array(scores, dtype=np.float32)
            
            for bbox, label, score in zip(boxes, labels, scores):
                res_json.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': bbox.tolist(),
                    'score': float(score)
                })
            
    save_filename = f'{model_name}-fold{fold}-e{epoch}-aug++.json'
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, 'w') as f:
        json.dump(res_json, f)


def run_wbf(
    root_path, json_paths, save_dir, save_filename, 
    iou_thr=0.44, skip_box_thr=0.43, weights=None
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    predictions = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            json_file = json.load(f)
            
        min_i = sorted(os.listdir('./data/test/'))[0]
        min_i = int(min_i.split('_')[-1].split('.')[0])
        max_i = sorted(os.listdir('./data/test/'))[-1]
        max_i = int(max_i.split('_')[-1].split('.')[0])
        
        prediction = {
            image_id: {'labels': [], 'boxes': [], 'scores': []}
            for image_id in range(min_i, max_i+1)
        }

        for annot in tqdm(json_file):
            img_name = f'test_{annot["image_id"]}.jpg'
            img_path = os.path.join(root_path, img_name)
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            
            x1, y1, w, h = annot['bbox']
            x2, y2 = x1+w, y1+h
            
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            bbox[[0,2]] = (bbox[[0,2]]/width).clip(min=0, max=1)
            bbox[[1,3]] = (bbox[[1,3]]/height).clip(min=0, max=1)

            prediction[annot['image_id']]['labels'].append(annot['category_id']) 
            prediction[annot['image_id']]['boxes'].append(bbox.tolist()) 
            prediction[annot['image_id']]['scores'].append(annot['score']) 
            
        predictions.append(prediction)
    
    wbf_predictions = []
    for image_id in tqdm(prediction.keys()):
        img_name = f'test_{image_id}.jpg'
        img_path = os.path.join(root_path, img_name)
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        boxes = [prediction[image_id]['boxes'] for prediction in predictions]
        labels = [prediction[image_id]['labels'] for prediction in predictions]
        scores = [prediction[image_id]['scores'] for prediction in predictions]
            
        boxes, scores, labels = weighted_boxes_fusion(
            boxes, scores, labels, weights=weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        boxes[:, [0,2]] = (boxes[:, [0,2]]*width).clip(min=0, max=width-1)
        boxes[:, [1,3]] = (boxes[:, [1,3]]*height).clip(min=0, max=height-1)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        boxes[:, 2] = widths
        boxes[:, 3] = heights

        for i, (bbox, score, label) in enumerate(zip(boxes, scores, labels)):      
            wbf_predictions.append({
                'image_id': int(image_id),
                'category_id': int(label),
                'bbox': bbox.tolist(),
                'score': float(score),
            })
            
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, 'w') as f:
        json.dump(wbf_predictions, f)
    
    
def get_submission(
    json_path, save_dir, save_filename, 
    submission_thr=0.5
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    
    results = {
        'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
        'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
    }
                
    for batch in tqdm(json_file):
        image_id = f'test_{batch["image_id"]}.json'
        
        if batch['score'] >= submission_thr:
            results['file_name'].append(image_id)
            results['class_id'].append(batch['category_id'])
            results['confidence'].append(batch['score'])
            
            x_min, y_min, w, h = batch['bbox']
            x_max, y_max = x_min + w, y_min + h
            results['point1_x'].append(x_min)
            results['point1_y'].append(y_min)
            results['point2_x'].append(x_max)
            results['point2_y'].append(y_min)
            results['point3_x'].append(x_max)
            results['point3_y'].append(y_max)
            results['point4_x'].append(x_min)
            results['point4_y'].append(y_max)
                    
    submission = pd.DataFrame(results)
    save_path = os.path.join(save_dir, save_filename)
    submission.to_csv(save_path, index=False)
    

def eval(ckpt_path, model_name, args, fold, epoch):
    data_module = EfficientDetDataModule(
        image_dir='./data',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        predict_transforms=get_predict_transforms(image_size=args.image_size)
    )

    model = EfficientDetModel(
        mode='eval',
        model_architecture=args.architecture_name,
        num_classes=4,
        image_size=args.image_size,
    )

    trainer = pl.Trainer(
        # gpus=[0, 1, 2, 3],
        gpus=1,
        # strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
    )

    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])
    # ckpt = torch.load("./weights/trained_effdet.pth", map_location='cuda:0')
    # model.load_state_dict(ckpt)

    outputs = trainer.predict(model, data_module)

    get_result(outputs, './results/', fold, epoch, model_name)


def main():
    convert_to_jpg('test', './ori_data/', './data/test')
        
    for root, dirs, files in os.walk('./weights'):
        if not dirs:
            ckpt_dir = root
            ckpt_name = files[0]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            
            args = get_args()
            model_name = ckpt_dir.split('\\')[1]
            
            if model_name == 'effdetd5':
                args.architecture_name = 'tf_efficientdet_d5'
            elif model_name == 'effdetd7i512':
                args.architecture_name = 'tf_efficientdet_d7'
                args.image_size = 512
                
            fold = int(ckpt_dir.split('\\')[-1])
            epoch = int(ckpt_name.split('-')[0].split('=')[-1])    
    
            eval(ckpt_path, model_name, args, fold, epoch)

    # ckpt_name = 'epoch=49-val_map50=0.01.ckpt'
    # ckpt_dir = './weights/effdetd5/3/'
    # ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    root_path = './data/test/'
    json_paths = sorted(glob('./results/*.json'))
    weights = [1., 1.1, 1., 1.1, 1., 1.2, 1.2, 1.2, 1.2]
    iou_thr = 0.55
    skip_box_thr = 0
    wbf_save_dir = './wbf_results/'
    wbf_save_filename = f'effdetd5+d7i512-fold0,1,2,3,4,0,1,2,3-wbf{weights}.json'
    
    json_dir = 'wbf_results'
    # json_dir = 'results'
    json_name = f'{wbf_save_filename.split(".json")[0]}'
    # json_name = 'effdetd7i512-fold3-e24-aug++'
    json_path = f'./{json_dir}/{json_name}.json'
    # save_path = f'./examples/test/effdetd5/wbf/0,1,2,3,4/{weights}'
    save_path = f'./examples/test/effdetd5+d7i512/wbf/0,1,2,3,4,0,1,2,3'
    # save_path = f'./examples/test/effdetd7i512/fold/3/'
    
    submission_thr = 0.22
        
    run_wbf(
        root_path, json_paths, 
        wbf_save_dir, wbf_save_filename,
        iou_thr, skip_box_thr, weights,
    )

    draw_bbox_result(
        root_path, json_path, save_path, 
        threshold=submission_thr
    )
    
    get_submission(
        json_path, save_dir='./submissions/', 
        save_filename=f'{json_name}-t{submission_thr}.csv', 
        submission_thr=submission_thr
    )  


if __name__ == '__main__':
    main()