import os
import cv2

'''
VOC数据对应
1: bicycle: 0: non-motor-vehicle
5: bus: 1: motor-vehicle
6: car: 1: motor-vehicle
13: motorbike: 0: non-motor-vehicle
14: person: 2: person

COCO数据对应
0: person: 2: person
1: bicycle: 0: non-motor-vehicle
2: car: 1: motor-vehicle
3: motorcycle: 0: non-motor-vehicle
5: bus: 1: motor-vehicle
7: truck: 1: motor-vehicle
'''

voc_dict = {'1': '0', '5': '1', '6': '1', '13': '0', '14': '2'}
coco_dict = {'0': '2', '1': '0', '2': '1', '3': '0', '5': '1', '7': '1'}


def transform_labels(labels, class_ids, dataset):
    final_labels = []
    for label in labels:
        label_id = label.split()[0]
        if label_id in class_ids:
            final_label = label.replace(label_id, dataset[label_id], 1)
            final_labels.append(final_label)
    return final_labels


def main(phase='train'):

    if phase == 'train':
        # image and label path
        vsp_path = 'vsp_object_detection_train'
        # get voc and coco path
        voc_path = '/mnt/data/user/lihan/voc/train.txt'
        coco_path = '/mnt/data/user/lihan/coco/trainvalno5k.txt'
        img_paths = 'train.txt'
    elif phase == 'test':
        # image and label path
        vsp_path = 'vsp_object_detection_val'
        # get voc and coco path
        voc_path = '/mnt/data/user/lihan/voc/2007_test.txt'
        coco_path = '/mnt/data/user/lihan/coco/val.txt'
        img_paths = 'val.txt'

    if not os.path.exists(vsp_path):
        os.mkdir(vsp_path)
        with open(os.path.join(vsp_path, 'classes.txt'), 'w') as f:
            f.write('non-motor-vechicle\n')
            f.write('motor-vechicle\n')
            f.write('person\n')

    all_path = {'voc_path': None, 'coco_path': None}
    with open(voc_path, 'r') as f:
        data = f.readlines()
        all_path['voc_path'] = data

    with open(coco_path, 'r') as f:
        data = f.readlines()
        all_path['coco_path'] = data

    # process data
    for key in all_path.keys():
        for img_name in all_path[key]:
            img_name = img_name.strip()

            if key == 'voc_path':
                label_name = img_name.replace('JPEGImages', 'labels').replace('jpg', 'txt')
                class_ids = ['1', '5', '6', '13', '14']
                dataset = voc_dict
            elif key == 'coco_path':
                label_name = img_name.replace('images', 'labels').replace('jpg', 'txt')
                class_ids = ['0', '1', '2', '3', '5', '7']
                dataset = coco_dict
            else:
                continue

            with open(label_name, 'r') as f:
                labels = f.readlines()
            final_labels = transform_labels(labels, class_ids, dataset)
            if len(final_labels) == 0:
                continue
            print('Processing image: {}'.format(img_name))
            src_img = cv2.imread(img_name)
            final_img_name = img_name.split('/')[-1]
            final_img_name = os.path.join(vsp_path, final_img_name)
            cv2.imwrite(final_img_name, src_img)
            with open(img_paths, 'a') as f:
                f.write(final_img_name + '\n')
            final_txt_name = final_img_name.replace('jpg', 'txt')
            with open(final_txt_name, 'w') as f:
                if key == 'voc_path':
                    f.writelines(final_labels)
                elif key == 'coco_path':
                    for final_label in final_labels:
                        f.write(final_label[:-2] + '\n')
                else:
                    continue


if __name__ == '__main__':
    for phase in ['train', 'test']:
        main(phase)
