from tqdm.notebook import tqdm
from glob import glob
import pandas as pd
import numpy as np
import itertools
import argparse
import json
import os

def get_id(x):
    x = os.path.basename(x)
    x, _ = os.path.splitext(x)
    return x

def get_name(x):
    x = os.path.basename(x)
    return x

def main(args):
    video_root = os.path.join(args.video_path, '**/*.mp4')
    segmentation_path = args.segmentation_path
    qna_path = args.qna_path

    df_videos = pd.DataFrame({'video_path':glob(video_root, recursive=True)})
    df_videos['video_name'] = df_videos['video_path'].map(get_name)
    df_segmentation_data = pd.DataFrame(json.load(open(segmentation_path)))
    df_qna_data = pd.DataFrame(json.load(open(qna_path)))

    df_common_dataset = pd.merge(df_videos, df_segmentation_data, on=['video_name'])
    df_common_dataset = pd.merge(df_common_dataset, df_qna_data, on=['video_id'])

    dataset_list = df_common_dataset['video_id'].unique()
    dataset_list_len = dataset_list.shape[0]

    np.random.shuffle(dataset_list)
    trainset_list = dataset_list[:int(dataset_list_len*0.8)]
    validset_list = dataset_list[int(dataset_list_len*0.8):int(dataset_list_len*0.9)]
    testset_list = dataset_list[int(dataset_list_len*0.9):]

    return {
        'train':trainset_list.tolist(),
        'validation':validset_list.tolist(),
        'test':testset_list.tolist()
        }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--segmentation_path', type=str, default=None)
    parser.add_argument('--qna_path', type=str, default=None)
    parser.add_argument('--common_split_list', type=str, default='common_dataset_list.json')
    args = parser.parse_args()
    
    if args.segmentation_path is None:
        args.segmentation_path = os.path.join(args.label_path, 'segmentation_data.json')
    if args.qna_path is None:
        args.qna_path = os.path.join(args.label_path, 'QNA_data.json')

    dataset_list = main(args)
    json.dump(dataset_list, open(args.common_split_list, 'w'), indent=4)
