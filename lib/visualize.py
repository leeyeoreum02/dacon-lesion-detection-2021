import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict


def get_colors(classes: List) -> Dict[str, tuple]:
    return {c: tuple(map(int, np.random.randint(0, 255, 3))) for c in classes}


def draw_bbox_submission(root_path, csv_path, save_path, n_images=100, threshold=0.5):
    df = pd.read_csv(csv_path)
    
    categories = {
        1: '01_ulcer',
        2: '02_mass',
        3: '04_lymph',
        4: '05_bleeding'
    }
    
    tmp = defaultdict(list)
    for _, row in tqdm(df.iterrows()):
        x1, y1 = row['point1_x'], row['point1_y']
        x2, y2 = row['point3_x'], row['point3_y']
        tmp[row['file_name']].append({
            'category_id': row['class_id'],
            'confidence': row['confidence'],
            'bbox': [x1, y1, x2, y2]
        })
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        
    colors = get_colors(categories.values())
    for image_id, annots in tqdm(list(tmp.items())[:n_images]):
        image_id = image_id.split('.')[0] + '.jpg'
        file_path = os.path.join(root_path, image_id)
        # print(file_path)
        
        image = cv2.imread(file_path)
        # image = cv2.imread(image_id)
        
        for a in annots:
            score = a['confidence']
            # print(score)
            if score >= threshold:
                label = categories[a['category_id']]
                x1, y1, x2, y2 = map(round, a['bbox'])
                
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
                (tw, th), _ = cv2.getTextSize(f'{label}: {score:.4f}', cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1-20), (x1+tw, y1), colors[label], -1)
                cv2.putText(image, f'{label}: {score:.4f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # cv2.putText(image, str(score)[:4], (x1+150, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # file_name = image_id.split('\\')[-1]
        cv2.imwrite(os.path.join(save_path, image_id), image)
    

def draw_bbox_result(root_path, json_path, save_path, n_images=100, threshold=0.5):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    
    categories = {
        1: '01_ulcer',
        2: '02_mass',
        3: '04_lymph',
        4: '05_bleeding'
    }
    
    tmp = defaultdict(list)
    for ann in tqdm(json_file):
        x1, y1, w, h = ann['bbox']
        x2, y2 = x1 + w, y1 + h
        image_id = f'test_{ann["image_id"]}'
        tmp[image_id].append({
            'category_id': ann['category_id'],
            'score': ann['score'],
            'bbox': [x1, y1, x2, y2]
        })
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        
    colors = get_colors(categories.values())
    for image_id, annots in tqdm(list(tmp.items())[:n_images]):
        image_id += '.jpg'
        file_path = os.path.join(root_path, image_id)
        # print(file_path)
        
        image = cv2.imread(file_path)
        # image = cv2.imread(image_id)
        
        for a in annots:
            # if image_id == 'test_200003.jpg' or image_id == 'test_200004.jpg':
            #     print('score:', a['score'])
            #     print('bbox:', list(map(round, a['bbox'])))
            #     print('label:', categories[a['category_id']])
            score = a['score']
            # print(score)
            if score >= threshold:
                label = categories[a['category_id']]
                x1, y1, x2, y2 = map(round, a['bbox'])
                
                cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
                (tw, th), _ = cv2.getTextSize(f'{label}: {score:.4f}', cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1-20), (x1+tw, y1), colors[label], -1)
                cv2.putText(image, f'{label}: {score:.4f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # cv2.putText(image, str(score)[:4], (x1+150, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # file_name = image_id.split('\\')[-1]
        cv2.imwrite(os.path.join(save_path, image_id), image)
    

def main():
    root_path = './data/test/'
    csv_path = './baseline.csv'
    json_path = './results/effdetd5-fold0-e41-aug++.json'
    # save_path = './examples/test/'
    save_path = './examples/test/effdetd5/0'

    draw_bbox_result(root_path, json_path, save_path, threshold=0)
    # draw_bbox_submission(root_path, csv_path, save_path)


if __name__ == '__main__':
    main()