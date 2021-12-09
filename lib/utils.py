import os
import json
import base64
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
from io import BytesIO
from PIL import Image
from collections import defaultdict


def convert_to_coco(
    root_path: os.PathLike, 
    save_path: os.PathLike
) -> None:
    """
        only for train dataset
    """
    res = defaultdict(list)
    json_paths = glob(os.path.join(root_path, 'train', '*.json'))
    
    categories = {
        '01_ulcer': 1,
        '02_mass': 2,
        '04_lymph': 3,
        '05_bleeding': 4
    }
    
    n_id = 0
    for json_path in tqdm(json_paths):
        with open(json_path, 'r') as f:
            tmp = json.load(f)
            
        image_id = int(tmp['file_name'].split('_')[-1][:6])
        res['images'].append({
            'id': image_id,
            'width': tmp['imageWidth'],
            'height': tmp['imageHeight'],
            'file_name': tmp['file_name'],
        })
        
        for shape in tmp['shapes']:
            box = np.array(shape['points'])
            x1, y1, x2, y2 = \
                min(box[:, 0]), min(box[:, 1]), max(box[:, 0]), max(box[:, 1])
            
            w, h = x2 - x1, y2 - y1
            
            res['annotations'].append({
                'id': n_id,
                'image_id': image_id,
                'category_id': categories[shape['label']],
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0,
            })
            n_id += 1
    
    for name, id in categories.items():
        res['categories'].append({
            'id': id,
            'name': name,
        })
        
    with open(save_path, 'w') as f:
        json.dump(res, f)
        
        
def convert_to_jpg(
    mode: str,
    root_path: os.PathLike,
    save_path: os.PathLike,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        
    json_paths = glob(os.path.join(root_path, mode, '*.json'))
    for json_path in tqdm(json_paths):
        with open(json_path, 'r') as f:
            tmp = json.load(f)
        image = BytesIO(base64.b64decode(tmp['imageData']))
        image = Image.open(image).convert('RGB')
        image.save(os.path.join(save_path, tmp['file_name'].split('.')[0])+'.jpg')