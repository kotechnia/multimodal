from preprocess.make_datasets import make_answer, make_answer_df
from subprocess import run
import pandas as pd
import argparse
import os

def main(args):
    print('Make dataset...')
    annotation_file = 'data/multimodal/annotation/annotation.csv'
    os.makedirs(os.path.dirname(annotation_file), exist_ok=True)

    answers = make_answer(args.annot_json)
    make_answer_df(answers, args.split_list, args.lang).to_csv(annotation_file, index=False, sep='\t')
    print('Complete!')

    if args.preprocess_video == 'y':
        print('Preprocessing videos : resnet...')
        run('''python preprocess/preprocess_features.py \
            --gpu_id 0 \
            --dataset multimodal \
            --video_path {} \
            --model resnet101 \
            --question_type action'''.format(args.video_path), shell=True)

        print('Preprocessing videos : resnext...')
        run('''python preprocess/preprocess_features.py \
              --dataset multimodal \
              --video_path {} \
              --model resnext101 \
              --image_height 112 \
              --image_width 112 \
              --question_type action'''.format(args.video_path), shell=True)

    print('Preprocessing annotations...')
    if args.lang=='ko':
        for opt, default in [('glove_pt', 'data/glove/glove.768d.ko.pkl'), ('token_type', 'transformer')]:
            if not vars(args)[opt]:
                vars(args)[opt] = default
    run('''python preprocess/preprocess_questions.py \
            --dataset multimodal \
            --glove_pt {} \
            --question_type action \
            --token_type {} \
            --tokenizer {}
            --by_video y \
            --mode train'''.format(args.glove_pt,
                                   args.token_type,
                                   args.tokenizer), shell=True)
    print('Complete!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_json', required=True)
    parser.add_argument('--split_list', required=True)
    parser.add_argument('--preprocess_video', choices=['y','n'], default='y')
    parser.add_argument('--video_path', )
    parser.add_argument('--lang', default='en')
    parser.add_argument('--glove_pt', default='data/glove/glove.840.300d.pkl')
    parser.add_argument('--token_type', choices=['nltk','transformers'], default='nltk')
    parser.add_argument('--tokenizer', default='monologg/koelectra-base-v3-discriminator')
    args = parser.parse_args()

    assert (args.preprocess_video=='y' and video_path) or (args.preprocess_video == 'n') ,"check your option 'preprocess_video' : If you want to preprocess the video, put option 'video_path'."

    main(args)
