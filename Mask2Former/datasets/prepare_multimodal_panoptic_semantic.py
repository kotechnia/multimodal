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
        panoptic = np.full((1920,1080,3),3, dtype=np.uint32)

    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
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
    print(id_map)

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
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


def split_dataset(root, categories):
    root = os.path.join(root, 'annotations')
    json_file = os.path.join(root, 'segmentation_data.json' )
    split_file = os.path.join(root, 'common_dataset_split.json')

    if not (os.path.exists(json_file) and os.path.exists(split_file)):
        return None

    json_data = json.load(open(json_file))
    video_keys = [video['video_id'] for video in json_data]
    split_list = json.load(open(split_file))
    split_list = {key : [video_keys.index(v_id) for v_id in value] for key, value in split_list.items()}

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
    parser.add_argument('--make_dataset', choices=['y', 'n'], default='n')
    args = parser.parse_args()


    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "multimodal")
    COCO_CATEGORIES = json.load(open(os.path.join(dataset_dir, 'annotations/categories.json'), encoding='utf-8-sig'))

    
    if args.make_dataset == 'y':
        split_dataset(dataset_dir, COCO_CATEGORIES)

    for s in ["test","val", "train"]:
        separate_coco_semantic_from_panoptic(
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "panoptic_{}".format(s)),
            os.path.join(dataset_dir, "panoptic_semseg_{}".format(s)),
            COCO_CATEGORIES,
        )
