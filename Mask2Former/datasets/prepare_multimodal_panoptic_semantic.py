#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import argparse
import pandas as pd
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image

#from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    
    if len(panoptic.shape) != 3 and len(np.unique(panoptic)) == 1:
        png_name = os.path.basename(input_panoptic)
        return png_name
#        panoptic = np.full((1920,1080,3),3, dtype=np.uint32)

    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)
    return "1"


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories, del_solid_color):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255

    with open(panoptic_json, encoding='utf-8-sig') as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            output_path = os.path.split(output)[0]
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    error_file = pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    if len(np.unique(error_file)) != 1 and del_solid_color:
        #print(np.unique(error_file))
        obj['annotations'] = [ ann for ann in obj['annotations'] if ann['file_name'] not in error_file]
        error_file = [name.replace('png', 'jpg') for name in error_file]
        obj['images'] = [ img for img in obj['images'] if img['file_Name'] not in error_file]

        json.dump(obj, open(panoptic_json, 'w'), indent=4)
    print("Finished. time: {:.2f}s".format(time.time() - start))


def split_dataset(root, json_file, split_file, categories):
    root = os.path.join(root, 'annotations')

    if not (os.path.exists(json_file) and os.path.exists(split_file)):
        return None

    json_data = json.load(open(json_file))
    video_keys = [video['video_id'] for video in json_data]
    split_list = json.load(open(split_file))
    split_list = {key : [video_keys.index(v_id) for v_id in value if v_id in video_keys] for key, value in split_list.items()}

    for mode in ['train', 'validation', 'test']:
        tmp_data = np.array(json_data)[split_list[mode]]
        imgs = pd.DataFrame([img for video in tmp_data for img in video['images']])
        anns = pd.DataFrame([ann for video in tmp_data for ann in video['annotations']] )

        imgs = imgs.drop_duplicates(subset='image_id', keep='last')
        anns = anns.drop_duplicates(subset='image_id', keep='last')


        if mode == 'validation':
            mode = 'val'

        json.dump( {'images' : imgs.to_dict('records'), 'annotations' : anns.to_dict('records'), 'categories' : categories} , open(os.path.join(root, 'panoptic_'+mode+'.json'),'w'), indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path',type=str)
    parser.add_argument('--split_list', type=str)
    parser.add_argument('--category_json', type=str)
    args = parser.parse_args()

    assert (args.annotation_path==None and args.split_list==None ) or (args.annotation_path!=None and args.split_list!=None)

    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "multimodal")
    if args.category_json:
        COCO_CATEGORIES = json.load(open(args.category_json))
    else:
        COCO_CATEGORIES = json.load(open(os.path.join(dataset_dir, 'annotations/categories.json')))

    if args.annotation_path:
        split_dataset(dataset_dir, args.annotation_path, args.split_list, COCO_CATEGORIES)


#    for s in ["test","val", "train"]:
#        if s!='train': del_solid_color = True
#        else: del_solid_color = False
#
#        separate_coco_semantic_from_panoptic(
#            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
#            os.path.join(dataset_dir, "panoptic_{}".format(s)),
#            os.path.join(dataset_dir, "panoptic_semseg_{}".format(s)),
#            COCO_CATEGORIES,
#            del_solid_color,
#        )
