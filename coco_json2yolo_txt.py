import os
import cv2
import json


stage = 'val2017'
label_file = '/media/F/COCO2017/annotations/instances_{}.json'.format(stage)

with open(label_file, 'r') as f:
    coco_json = json.load(f)

result_path = '/media/F/object2017'
if not os.path.exists(result_path):
    os.makedirs(result_path)
for annotation in coco_json['annotations']:
    image_name = '/media/F/COCO2017/{}/{:012d}.jpg'.format(stage, annotation['image_id'])
    print('Processing {} ...'.format(image_name))
    src_img = cv2.imread(image_name)
    if src_img is None:
        print('Image name error!')
    cat = annotation['category_id']
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    object_name = coco_json['categories'][cat]['name']
    img_dir = os.path.join(result_path, stage)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    img_name = os.path.join(img_dir, '{:012d}.jpg'.format(annotation['image_id']))
    cv2.imwrite(img_name, src_img)
    height, width, _ = src_img.shape
    txt_name = img_name.replace('jpg', 'txt')
    xt, yt, w, h = annotation['bbox']
    xc = xt + w / 2
    yc = yt + h / 2
    xc, yc, w, h = xc / width, yc / height, w / width, h / height
    with open(txt_name, 'a') as ft:
        ft.write('{} {} {} {} {}\n'.format(cat, xc, yc, w, h))
