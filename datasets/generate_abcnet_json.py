#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import sys
import cv2
import numpy as np
from shapely.geometry import *

# Desktop Latin_embed.
vn_dict = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
'A','Â','Ă','À','Á','Ả','Ã','Ạ','Ầ','Ấ','Ẩ','Ẫ','Ậ','Ằ','Ắ','Ẳ','Ẵ','Ặ',
'B','C','D','Đ','E','Ê','È','É','Ẻ','Ẽ','Ẹ','Ề','Ế','Ể','Ễ','Ệ',
'F','G','H','I','Ì','Í','Ỉ','Ĩ','Ị','J','K','L','M','N',
'O','Ô','Ơ','Ò','Ó','Ỏ','Õ','Ọ','Ồ','Ố','Ổ','Ỗ','Ộ','Ờ','Ớ','Ở','Ỡ','Ợ','P','Q','R','S','T',
'U','Ư','Ù','Ú','Ủ','Ũ','Ụ','Ừ','Ứ','Ử','Ữ','Ự','V','W','X','Y','Ỳ','Ý','Ỷ','Ỹ','Ỵ','Z','[','\\',']','^','_','`',
'a','â','ă','à','á','ả','ã','ạ','ầ','ấ','ẩ','ẫ','ậ','ằ','ắ','ẳ','ẵ','ặ',
'b','c','d','đ','e','ê','è','é','ẻ','ẽ','ẹ','ề','ế','ể','ễ','ệ',
'f','g','h','i','ì','í','ỉ','ĩ','ị','j','k','l','m','n',
'o','ô','ơ','ò','ó','ỏ','õ','ọ','ồ','ố','ổ','ỗ','ộ','ờ','ớ','ở','ỡ','ợ','p','q','r','s','t',
'u','ư','ù','ú','ủ','ũ','ụ','ừ','ứ','ử','ữ','ự','v','w','x','y','ỳ','ý','ỷ','ỹ','ỵ','z','{','|','}','~']

# if len(sys.argv) < 3:
#   print("Usage: python convert_to_detectron_json.py root_path phase split")
#   print("For example: python convert_to_detectron_json.py data train 100200")
#   exit(1)
# root_path = sys.argv[1]
# phase = sys.argv[2]
# split = int(sys.argv[3])
root_path = './'
dataset = {
    'licenses': [],
    'info': {},
    'categories': [],
    'images': [],
    'annotations': []
}
with open(os.path.join(root_path, 'classes.txt')) as f:
  classes = f.read().strip().split()
for i, cls in enumerate(classes, 1):
  dataset['categories'].append({
      'id': i,
      'name': cls,
      'supercategory': 'beverage',
      'keypoints': ['mean',
                    'xmin',
                    'x2',
                    'x3',
                    'xmax',
                    'ymin',
                    'y2',
                    'y3',
                    'ymax',
                    'cross']  # only for BDN
  })


def get_category_id(cls):
  for category in dataset['categories']:
    if category['name'] == cls:
      return category['id']


indexes = sorted([f.split('.')[0]
                   for f in os.listdir(os.path.join(root_path, 'train/labels'))])

# if phase == 'train':
#   indexes = [line for line in _indexes if int(
#       line) >= split]  # only for this file
# else:
#   indexes = [line for line in _indexes if int(line) <= split]
j = 1
for index in indexes:
  # if int(index) >3: continue
  print('Processing: ' + index)
  im = cv2.imread(os.path.join(root_path, 'train/images/') + index + '.jpg')
  height, width, _ = im.shape
  dataset['images'].append({
      'coco_url': '',
      'date_captured': '',
      'file_name': index + '.jpg',
      'flickr_url': '',
      'id': index.split('_')[1],
      'license': 0,
      'width': width,
      'height': height
  })
  anno_file = os.path.join(root_path, 'train/labels/') + index + '.txt'

  with open(anno_file, encoding="utf8") as f:
    lines = [line for line in f.readlines() if line.strip()]
    for i, line in enumerate(lines):
      pttt = line.strip().split(',')
      parts = pttt[:8]
      parts = [round(float(x)) for x in parts]
      ct = pttt[-1].strip()

      cls = 'text'
      #segs = [float(kkpart) for kkpart in parts[:16]]  
      x0 = parts[0]
      y0 = parts[1]
      x3 = parts[2]
      y3 = parts[3]
      x1 = round((x3+2*x0)/3)
      y1 = round((y3+2*y0)/3)
      x2 = round((2*x3+x0)/3)
      y2 = round((2*y3+y0)/3)

      x4 = parts[4]
      y4 = parts[5]
      x7 = parts[6]
      y7 = parts[7]
      x5 = round((x7+2*x4)/3)
      y5 = round((y7+2*y4)/3)
      x6 = round((2*x7+x4)/3)
      y6 = round((2*y7+y4)/3)
      segs = [x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7]
      xmin = min([parts[0],parts[2],parts[4],parts[6]])
      ymin = min([parts[1],parts[3],parts[5],parts[7]])
      xmax = max([parts[0],parts[2],parts[4],parts[6]])
      ymax = max([parts[1],parts[3],parts[5],parts[7]])
      width = max(0, xmax - xmin + 1)
      height = max(0, ymax - ymin + 1)
      if width == 0 or height == 0:
        continue

      max_len = 100
      recs = [len(vn_dict)+1 for ir in range(max_len)]
      
      ct =  str(ct)
      print('rec', ct)
      
      for ix, ict in enumerate(ct):        
        if ix >= max_len: continue
        if ict in vn_dict:
            recs[ix] = vn_dict.index(ict)
        else:
          recs[ix] = len(vn_dict)

      dataset['annotations'].append({
          'area': width * height,
          'bbox': [xmin, ymin, width, height],
          'category_id': get_category_id(cls),
          'id': j,
          'image_id': index.split('_')[1],
          'iscrowd': 0,
          'bezier_pts': segs,
          'rec': recs
      })
      j += 1
folder = os.path.join(root_path, 'train/annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'train/annotations/{}.json'.format('train'))
with open(json_name, 'w') as f:
  json.dump(dataset, f)
