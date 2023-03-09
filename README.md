# multimodal

## Index
- [Common](#common)
- [Mask2Former - Panoptic Segmentation model](#mask2former)
- [HCRN  - QnA model](#hcrn)
- [LICENSE](#license)

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


## Common
> Mask2Former와 HCRN에서 공통된 훈련, 검증, 시험 세트를 사용하기 위해 데이터를 분할합니다.
```bash
python prepare.py \
--video_path data/videos \
--segmentation_path data/annotations/segmentation_data.json \
--qna_path data/annotations/QNA_data.json \
--common_split_list common_split_list.json
```
<br/>

## Mask2Former

model : Mask2Former  
config : configs/multimodal/config_multimodal.yaml


❏ maks2former의 구조는 다음과 같습니다.
```
📂mask2former
├─ 📂configs
│   └─ 📂multimodal
├─ 📂datasets
│   └─ 📂multimodal
│       ├─ 📂annotations
│       │  ├─ 📄categories.json
│       │  ├─ 📄train.json
│       │  ├─ 📄val.json
│       │  └─ 📄test.json
│       ├─ 📂panoptic_train
│       ├─ 📂panoptic_val
│       ├─ 📂panoptic_test
│       ├─ 📂train
│       ├─ 📂val
│       └─ 📂test
├─ 📄README.md
├─ 📄LICENSE
├─ 📄predict.py
├─ 📄requirements.txt
└─ 📄train_net.py
```
<br>
❏ 사용 라이브러리 및 프로그램입니다. 
<br> 


**detectron2**

```
$ git clone https://github.com/facebookresearch/detectron2.git
$ pip install -e detectron2
```

```
$ sed -i s/"int(ann\\['image_id'\\])"/"ann['image_id']"/g detectron2/detectron2/data/datasets/coco_panoptic.py 
```

**panopticapi**
```
$ git clone https://github.com/cocodataset/panopticapi.git
$ pip install -e panopticapi
```

**mask2former**
```
$ cd mask2former
$ pip install -r requirements.txt
$ cd mask2former/modeling/pixel_decoder/ops
$ sh make.sh
$ cd -
```
<br/>

### 데이터 전처리 방법 (예시)
```bash
$ source prepare_multimodal_dataset.sh \
../data/videos/video_frames \
../data/annotations/labels \
../data/annotations/segmentation_data.json \
../data/annotations/categories.json \
../common_split_list.json
```

### 실행 방법 (예시)

❏ 훈련 방법입니다.
```
$ python train_net.py  \
--config-file configs/multimodal/config_multimodal.yaml \
--num-gpus 2 \
SOLVER.IMS_PER_BATCH 2 \
OUTPUT_DIR ./multimodal
```

❏ 평가 방법입니다.
```
$ python train_net.py  \
--config-file multimodal/config.yaml
--num-gpus 2 \
--eval-only \
MODEL.WEIGHTS multimodal/model_final.pth \
DATASETS.EVAl multimodal_2022_test_dataset_panoptic 
```
<br>
<details>
    <summary>❏  original github & paper</summary>
    <p>github : <a href='https://github.com/facebookresearch/Mask2Former'>Mask2Former</a>
    <p>paper : <a href='https://arxiv.org/pdf/2112.01527.pdf'>arXiv:2112.01527</a>
</details>

---

## HCRN


model : HCRN  
config (한글) : configs/multimodal_qa_action_ko.yml  
config (영어) : configs/multimodal_qa_action_en.yml


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


**한글**  
```bash
$ python preprocess/make_datasets.py \
--annot_json ../data/annotations/QNA_data.json \
--split_list ../common_split_list.json \
--lang ko

$ python preprocess/preprocess_features.py \
--gpu_id 0 \
--dataset multimodal \
--video_path ../data/videos \
--model resnet101 \
--question_type action

$ python preprocess/preprocess_features.py \
--dataset multimodal \
--video_path ../data/videos \
--model resnext101 \
--image_height 112 \
--image_width 112 \
--question_type action

$ python preprocess/preprocess_questions.py \
--dataset multimodal \
--glove_pt data/glove/glove.768d.ko.pkl \
--question_type action \
--token_type transformers \
--by_video y \
--mode train
```
<br/>


**영어**  
```bash
# config file => train.word_dim: 300
import nltk
nltk.download('punkt')
```
```bash
$ python preprocess/make_datasets.py \
--annot_json ../data/annotations/QNA_data.json \
--split_list ../common_split_list.json \
--lang en

$ python preprocess/preprocess_features.py \
--gpu_id 0 \
--dataset multimodal \
--video_path ../data/videos \
--model resnet101 \
--question_type action

$ python preprocess/preprocess_features.py \
--dataset multimodal \
--video_path ../data/videos \
--model resnext101 \
--image_height 112 \
--image_width 112 \
--question_type action

$ python preprocess/preprocess_questions.py \
--dataset multimodal \
--glove_pt data/glove/glove.840.300d.pkl \
--question_type action \
--token_type transformers \
--by_video y \
--mode train
```
<br/>

### 실행 방법 (예시)
**한글**  
❏ 훈련 방법입니다.
```bash
$ python train.py --cfg configs/multimodal_qa_action_ko.yml
```

❏ 평가 방법입니다.
```bash
$ python validate.py --cfg configs/multimodal_qa_action_ko.yml
```
<br/>

**영어**  
❏ 훈련 방법입니다.
```bash
$ python train.py --cfg configs/multimodal_qa_action_en.yml
```

❏ 평가 방법입니다.
```bash
$ python validate.py --cfg configs/multimodal_qa_action_en.yml
```



<br>
<details>
    <summary>❏  original github & paper</b></summary>
    <p>github : <a href='https://github.com/thaolmk54/hcrn-videoqa'>HCRN</a>
    <p>paper : <a href='https://arxiv.org/pdf/2002.10698.pdf'>arXiv:2002.10698</a>
</details>

---

## License
> The license for this repository is based on the MIT license.   
> If the module has a "LICENSE" file, that license is applied.
