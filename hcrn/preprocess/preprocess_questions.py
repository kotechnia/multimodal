import argparse
import random
import json
import pandas as pd
import numpy as np
import os

from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa','multimodal'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--token_type', choices=['transformers', 'nltk'], default='nltk')
    parser.add_argument('--tokenizer')
    parser.add_argument('--by_video', choices=['y', 'n'], default='n')
    parser.add_argument('--annot_json')
    parser.add_argument('--lang', choices=['ko', 'en'])
    parser.add_argument('--split_list')

    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)



    if args.dataset == 'tgif-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/tgif-qa/csv/{}_{}_question.csv'
        args.output_pt = 'data/tgif-qa/{}/tgif-qa_{}_{}_questions.pt'
        args.vocab_json = 'data/tgif-qa/{}/tgif-qa_{}_vocab.json'
        # check if data folder exists
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msrvtt/annotations/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msvd/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)
    elif args.dataset == 'multimodal':
        args.annotation_file = 'data/multimodal/annotation/annotation.csv'
        args.output_pt = 'data/multimodal/features/multimodal/{}/multimodal_{}_{}_questions.pt'
        args.vocab_json = 'data/multimodal/features/multimodal/{}/multimodal_{}_vocab.json'
        # check if data folder exists
        #if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
        #    os.makedirs('data/tgif-qa/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            
            if args.split_list and args.annot_json and args.lang:
                df = make_answer(args.annot_json)
                make_answer_df(df, args.split_list, args.lang).to_csv(args.annotation_file, index=False, sep='\t')


            tgif_qa.process_questions_mulchoices(args)
