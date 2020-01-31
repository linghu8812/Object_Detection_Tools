#!/usr/bin/env python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import os
import sys
import re
import glob
import subprocess
from pathlib import Path


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate results over mAP range')

    # required
    parser.add_argument('--weights_folder', dest='weights_folder', type=str)
    parser.add_argument('--data_file', dest='data_file', type=str)
    parser.add_argument('--cfg_file', dest='cfg_file', type=str)

    parser.add_argument('--lib_folder', dest='lib_folder', default='', type=str)
    parser.add_argument('--gpu_id', dest='gpu_id', default='0', type=str)
    parser.add_argument('--min_weight_id', dest='min_weight_id', default=700, type=int)
    # does both metrics now
    # parser.add_argument('--metric', dest='metric', default='iou', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def file_to_weight_id(path):
    wid = None
    try:
        m = re.search('(\d+).weights', path)
        if m is None:
            m = re.search('(final)\.weights', path)
        wid = m.group(1)
    except AttributeError as ae:
        print("{}: {}".format(path, ae))
        raise ae
    return wid


def int_or_max(int_or_final):
    if int_or_final == 'final':
        return sys.maxint
    return int(int_or_final)


def main():
    """
    python scripts/voc_all_map.py --weights_folder [folder with weights] --data_file [whatever.data] --cfg_file [whatever.cfg]
    """
    with open('5k.txt', 'r') as f:
        img_files = f.read().splitlines()
    imgIds = [int(Path(x).stem.split('_')[-1]) for x in img_files]
    args = parse_args()
    if not args.weights_folder:
        raise str('you must pass a --weights_folder')
    weights_folder_name = list(filter(None, args.weights_folder.split('/')))[-1]
    output_path = "results/coco_results_{}".format(weights_folder_name)
    print("Output to: '{}'".format(output_path))
    try:
        os.makedirs(output_path)
    except OSError as ose:
        print("warning: {}".format(ose))

    all_weights_files = glob.glob(os.path.join(args.weights_folder, '*.weights'))
    # reverse sorted
    all_weights_files = sorted(all_weights_files, key=lambda a: int_or_max(file_to_weight_id(a)) > 0 or 1, reverse=True)
    print("Processing {}".format('\n'.join(all_weights_files)))

    visited = set()
    map_results_path = os.path.join(args.weights_folder, 'map.txt')
    # skip already visited
    if os.path.isfile(map_results_path):
        with open(map_results_path, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) < 1:
                    continue
                rows = line.split(',')
                # mteric, weight_id
                visited.add((rows[0], int_or_max(rows[1])))
    print("Skipping already visited {}".format(visited))

    for i, weights_file in enumerate(all_weights_files):
        weight_id = int_or_max(file_to_weight_id(weights_file))
        if weight_id < args.min_weight_id:
            continue
        if ('val2014-iou', weight_id) in visited:
            continue
        weights_path = os.path.dirname(weights_file)

        weights_output_paths = dict()
        year = 'val2014'
        weights_output_paths[year] = os.path.join(output_path, str(weight_id), year)
        res_file = os.path.join(weights_output_paths[year], 'coco_results.json')
        print("weights output to: '{}'".format(res_file))
        try:
            os.makedirs(weights_output_paths[year])
        except OSError as ose:
            print("warning: {}".format(ose))

        if os.path.isfile(res_file) and os.path.getsize(res_file) > 0:
            print("skipping generation of populated results file '{}'".format(res_file))
        else:
            ldlib = 'LD_LIBRARY_PATH={}'.format(args.lib_folder) if args.lib_folder else ''
            gpu = '-i {}'.format(args.gpu_id) if args.gpu_id else ''
            # date_file_with_year = "{}{}.data".format(args.data_file.split('.data')[0],
            #                                          (".{}".format(year) if len(year) else year))
            cmd = "{} ./darknet detector valid {} {} {} {} -prefix {}".format(ldlib, args.data_file,
                                                                              args.cfg_file, weights_file, gpu,
                                                                              weights_output_paths[year])
            print("running '{}'".format(cmd))
            retval = 0
            callerr = False
            try:
                retval = subprocess.call(cmd, shell=True)
            except OSError as ose:
                print("OSError: '{}'".format(ose))
                callerr = True
            print("{} finished with val {}".format(cmd, retval))

            cmd = "mv results/coco_results.json {}".format(weights_output_paths[year])
            print(cmd)
            if os.WEXITSTATUS(os.system(cmd)) != 0:
                assert "{} failed".format(cmd)
            print("./move files complete")

            sys.stdout.flush()
            if retval != 0 or callerr:
                raise Exception("'{}' failed".format(cmd))

            print("darknet run {}' complete".format(cmd))

        if len(weights_output_paths.items()) == 0:
            print("no weights_output_paths, breaking")
            break
        data_dir = '/home/linghu8812/data/coco'
        ann_file = '%s/annotations/instances_val2014.json' % data_dir
        print("loading {}".format(ann_file))

        # for metric in ['iou', 'giou']:
        metric = 'val2014-iou'
        if (metric, weight_id) in visited:
            continue
        year = 'val2014'

        print('Evaluating detections with {}'.format(metric))
        mAP_analysis = [','.join([metric, 'mAP'])]

        to_load = "{}/coco_results.json".format(weights_output_paths[year])
        cocoGt = COCO(ann_file)  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes(to_load)  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP = cocoEval.stats[0]
        mAP_5 = cocoEval.stats[1]
        mAP_75 = cocoEval.stats[2]
        mAP_S = cocoEval.stats[3]
        mAP_M = cocoEval.stats[4]
        mAP_L = cocoEval.stats[5]

        mAR = cocoEval.stats[6]
        mAR_5 = cocoEval.stats[7]
        mAR_75 = cocoEval.stats[8]
        mAR_S = cocoEval.stats[9]
        mAR_M = cocoEval.stats[10]
        mAR_L = cocoEval.stats[11]

        one = [metric, weight_id, mAP, mAP_5, mAP_75, mAP_S, mAP_M, mAP_L, mAR, mAR_5, mAR_75, mAR_S, mAR_M, mAR_L]
        mAP_analysis.append(','.join([str(o) for o in one]))
        results_path = os.path.join(weights_path, '{}-{}.txt'.format(weight_id, metric))
        print("Writing: '{}' and '{}'".format(results_path, map_results_path))
        with open(results_path, 'w') as f:
            f.write('\n'.join(mAP_analysis))

        # read/write for insertion sort
        # per-line format is ['giou|iou', weight file id, mean, 0.5..0.95 iou]
        reslines = []
        inserted = False
        linetoinsert = ','.join([str(o) for o in one])
        print("inserting: {}".format(linetoinsert))
        if os.path.isfile(map_results_path):
            with open(map_results_path, 'r') as f:
                for line in f.readlines():
                    if len(line.strip()) < 1:
                        continue
                    cols = line.split(',')
                    if (not inserted) and int(cols[1]) > weight_id:
                        reslines.append(linetoinsert)
                        inserted = True
                    reslines.append(line)
        else:
            reslines.append(linetoinsert)
            inserted = True
        if not inserted:
            reslines.append(linetoinsert)
        with open(map_results_path, 'w') as f:
            f.write('\n'.join([l.strip() for l in reslines]))


if __name__ == '__main__':
    main()
