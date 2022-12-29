# multimodal

## Index
- Mask2Former - Panoptic Segmentation model
- HCRN  - QnA model

## Contents

❏ 테스트 시스템 사양은 다음과 같습니다.
```
Ubuntu 20.04   
Python 3.8.10 
Torch 1.9.0+cu111 
CUDA 11.1
cuDnn 8.2.0   
```
<br>


❏ 각 모델 별 설명입니다.

## Mask2Former

model : Mask2Former  
config : configs/multimodal/config_multimodal.yaml


❏ maks2former의 구조는 다음과 같습니다.
```
📂mask2former
├─ 📂configs
│   ├─ 📂coco
│   └─ 📂multimodal
├─ 📂datasets
│   ├─ 📂multimodal
│   │   ├─ 📂annotations
│   │   │  ├─ 📄categories.json
│   │   │  ├─ 📄train.json
│   │   │  ├─ 📄val.json
│   │   │  └─ 📄test.json
│   │   ├─ 📂panoptic_train
│   │   ├─ 📂panoptic_val
│   │   ├─ 📂panoptic_test
│   │   ├─ 📂train
│   │   ├─ 📂val
│   │   └─ 📂test
│   └─ 📄prepare_coco_semantic_annos_from_panoptic_annos.py
├─ 📂demo
├─ 📂demo_video
├─ 📂mask2former
├─ 📂mask2former_video
├─ 📂tools
├─ 📄README.md
├─ 📄LICENSE
├─ 📄predict.py
├─ 📄requirements.txt
├─ 📄train_net.py
└─ 📄train_net_video.py
```
<br>
❏ 사용 라이브러리 및 프로그램입니다. 
<br> 


**detectron2**

```
$ git clone https://github.com/facebookresearch/detectron2.git
$ cd detectron2
$ pip install -e .
$ cd -
```

```
$ sed -i s/"int(ann\\['image_id'\\])"/"ann['image_id']"/g 
```

**panopticapi**
```
$ git clone https://github.com/cocodataset/panopticapi.git
$ cd panopticapi
$ pip install -e .
$ cd -
```

**mask2former**
```
$ cd mask2former
$ pip install -r requirements.txt
```
### 데이터 전처리 방법 (예시)
```bash
python datasets/prepare_multimodal_panoptic_semantic.py --make_dataset y
```

### 실행 방법 (예시)

❏ 훈련 방법입니다.
```
python train_net.py  \
--config-file configs/multimodal/config_multimodal.yaml \
--num-gpus 2 \
SOLVER.IMS_PER_BATCH 2 \
OUTPUT_DIR ./<output_dir name>
```

❏ 평가 방법입니다.
```
python train_net.py  \
--config-file <output_dir name>/config.yaml
--num-gpus 2 \
--eval-only \
MODEL.WEIGHTS <output_dir name>/checkpoint_file \
DATASETS.EVAl <dataset_name>   # multimodal_2022_test_dataset \ 
OUTPUT_DIR ./<output_dir name>
```
<br>
<details>
    <summary>❏  original github & paper</summary>
    <p>github : <a href='https://github.com/facebookresearch/Mask2Former'>Mask2Former</a>
    <p>paper : <a href='https://arxiv.org/pdf/2112.01527.pdf'>arXiv:2112.01527</a>
</details>

---
---

## HCRN


model : HCRN  
config : configs/multimodal_qa_action.yml


❏ HCRN의 구조는 다음과 같습니다.
```
📂cla
├─ 📂configs
│   └─ 📄multimodal_qa_action.yml
├─ 📂data
│   ├─ 📂glove
│   └─ 📂multimodal
│       ├─ 📂annotation
│       │   └─ 📄annotation.csv
│       └─ 📂video
├─ 📂model
├─ 📂preprocess
├─ 📄.gitignore
├─ 📄CRNUnit.png
├─ 📄DataLoader.py
├─ 📄LICENSE
├─ 📄README.md
├─ 📄config.py
├─ 📄overview.png
├─ 📄requirements.txt
├─ 📄train.py
├─ 📄utils.py
└─ 📄validate.py
```
<br>

### 데이터 전처리 방법 (예시)

❏ 비디오 전처리 방법입니다.
```bash
python preprocess/preprocess_features.py \
--gpu_id 0 \
--dataset multimodal \
--model resnet101 \
--question_type action

python preprocess/preprocess_features.py \
--dataset multimodal \
--model resnext101 \
--image_height 112 \
--image_width 112 \
--question_type action
```

❏ annotation 전처리 방법입니다.
<br>


**한글**
```bash
# config file => train.word_dim: 768
# huggingface의 transformers 모듈에 맞는 tokenizer
# => default: 'monologg/koelectra-base-v3-discriminator' 
# tokenizer 변경 시, 그에 맞게 glove_pt도 변경 필요
```

```bash
python preprocess/preprocess_questions.py \
--dataset multimodal \
--glove_pt data/glove/glove.768d.ko.pkl \   
--question_type action \
--token_type transformers \                 
--tokenizer tokenizer/my_tokenizer \
--by_video y \
--annot_json data/multimodal/annotation/QNA.json \
--split_list data/multimodal/annotation/common_dataset_split.json \
--lang ko \
--mode train
```


**영문**
```bash
# config file => train.word_dim: 300
import nltk
nltk.download('punkt')
```
```bash
python preprocess/preprocess_questions.py \
--dataset multimodal \
--glove_pt data/glove/glove.840.300d.pkl \
--question_type action \
--by_video y \
--annot_json data/multimodal/annotation/QNA.json \
--split_list data/multimodal/annotation/common_dataset_split.json \
--lang en \
--mode train
```

### 실행 방법 (예시)

❏ 훈련 방법입니다.
```bash
python train.py --cfg configs/multimodal_qa_action.yml
```

❏ 평가 방법입니다.
```bash
python validate.py --cfg configs/multimodal_qa_action.yml
```
<br>
<details>
    <summary>❏  original github & paper</b></summary>
    <p>github : <a href='https://github.com/thaolmk54/hcrn-videoqa'>HCRN</a>
    <p>paper : <a href='https://arxiv.org/pdf/2002.10698.pdf'>arXiv:2002.10698</a>
</details>

