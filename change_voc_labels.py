import os
import cv2


classes = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]

work_path = '/media/F/coco_voc'
for folder in ['images', 'labels']:
    folder_name = os.path.join(work_path, folder)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

with open('coco_voc_train.txt', 'w') as fw:
    for year in ['2007', '2012']:
        for root, dirs, files in os.walk('/home/linghu8812/data/voc/VOCdevkit/VOC{}/labels'.format(year), topdown=False):
            for name in files:
                label_file = os.path.join(root, name)
                if 'difficult' in label_file:
                    continue
                image_file = label_file.replace('labels', 'JPEGImages').replace('txt', 'jpg')
                print('Processing {} ...'.format(image_file))
                src_img = cv2.imread(image_file)
                rst_img_file = os.path.join('/media/F/coco_voc/images', image_file.split('/')[-1])
                fw.write(rst_img_file + '\n')
                cv2.imwrite(rst_img_file, src_img)
                with open(label_file, 'r') as fr:
                    labels = fr.readlines()
                with open(os.path.join('/media/F/coco_voc/labels', label_file.split('/')[-1]), 'w') as fl:
                    for label in labels:
                        index = label.split(' ')[0]
                        label = label.replace(index, str(classes[int(index)]))
                        fl.write(label)
                a = 0

a = 0
