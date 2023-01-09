import argparse

import pandas as pd
import numpy as np
import random
import json
import os

def make_answer(json_file):
    data = json.load(open(json_file))
    alls = []
    for video in data:
        video_id = video['video_id']
        video_name = video['video_name']

        for qna in video['qna_info']:
            tmp_dict = {'video_id' : video_id, 'video_name':video_name}
            tmp_dict.update(qna)
            alls.append(tmp_dict)
    df = pd.DataFrame(alls)
    
    df['answer_ko'] = np.where(df['ko_answer'].str.contains('^[네예][ \.\,]'), '네',
                      np.where(df['ko_answer'].str.contains('^[아이]니[오요]'), '아니요', df['ko_answer']))
    df['answer_en'] = np.where(df['en_answer'].str.contains('^[Yy]e[sp]'), 'Yes',
                      np.where(df['en_answer'].str.contains('^[Nn]o[a-z \,\.\?]+(?!-)'), 'No', df['en_answer']))
    df['answer_en'] = np.where(df['answer_ko'] == '네', 'Yes',
                      np.where(df['answer_ko'] == '아니요', 'No', df['answer_en']))
    df['answer_ko'] = np.where(df['answer_en'] == 'Yes', '네',
                      np.where(df['answer_en'] == 'No', '아니요', df['answer_ko']))
    
    return df
    

def make_answer_df(json_file, split_json, lang):
    split_list = json.load(open(split_json))

    alls = []
    if lang == 'ko':
        choice = ['네', '아니요']
    else:
        choice = ['Yes', 'No']

    for mode in ['train','validation','test']:
        df_t = df[df.video_id.isin(split_list[mode])]

        answers = df_t[f'answer_{lang}'].unique()

        for i in range(len(df_t)):
            tmp_df = df_t.iloc[i]
            origin = tmp_df[f'answer_{lang}']

            if origin in choice: answer = [choice[0], choice[1]]
            else: answer = [choice[0], choice[1], origin]


            while len(answer) != 5:
                candidate_a = random.choice(answers)
                if not candidate_a in answer: answer.append(candidate_a)

            random.shuffle(answer)

            tmp_dict = {'gif_name' : tmp_df['video_name'], 'vid_id': tmp_df['video_id'], 'key': tmp_df['video_id'],  'question': tmp_df[f'{lang}_question'], 'answer':answer.index(origin), 'mode':mode}
            tmp_dict.update({ f'a{i+1}' : value for i, value in enumerate(answer)})
            alls.append(tmp_dict)
            
    return pd.DataFrame(alls)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='multimodal', choices=['multimodal'], type=str)    
    parser.add_argument('--annot_json')
    parser.add_argument('--lang', choices=['ko', 'en'])
    parser.add_argument('--split_list')
    args = parser.parse_args()

    if args.dataset == 'multimodal':
        print('MAKE DATASET...')
        args.annotation_file = 'data/multimodal/annotation/annotation.csv'
        os.makedirs(os.path.dirname(args.annotation_file), exist_ok=True)
        df = make_answer(args.annot_json)
        make_answer_df(df, args.split_list, args.lang).to_csv(args.annotation_file, index=False, sep='\t')
        print('COMPLETE!')
    else:
        print('only for multimodal dataset!')
