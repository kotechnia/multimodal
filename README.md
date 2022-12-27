# multimodal
model : Mask2Former  
config : maskformer2_swin_large_IN21k_384_bs16_100ep.yaml


❏ multimodal의 구조는 다음과 같습니다.
```
📂mask2former
├─ 📂configs


📂mask2former
├─ 📂configs
│   ├─ 📂coco
│       ├─ 📂instance-segmentation
│       └─ 📂panoptic-segmentation
│           ├─ 📂swin
│           │   └─ 📄maskformer2_swin_large_IN21k_384_bs16_100ep.yaml
│           ├─ 📄Base-COCO-PanopticSegmentation.yaml
│           ├─ 📄maskformer2_R101_bs16_50ep.yaml
│           └─ 📄maskformer2_R50_bs16_50ep.yaml
├─ 📂datasets
│   ├─ 📂coco
│   │   ├─ 📂annotations
│   │   │  ├─ 📄panoptic_train2017.json
│   │   │  └─ 📄panoptic_val2017.json
│   │   ├─ 📂panoptic_train2017
│   │   ├─ 📂panoptic_val2017
│   │   ├─ 📂train2017
│   │   └─ 📂val2017
│   ├─ 📄README.md
│   └─ 📄prepare_coco_semantic_annos_from_panoptic_annos.py
├─ 📂demo
├─ 📂demo_video
├─ 📂mask2former
├─ 📂mask2former_video
├─ 📂tools
├─ 📄LICENSE
├─ 📄predict.py
├─ 📄requirements.txt
├─ 📄train_net.py
└─ 📄train_net_video.py
```

❏ 테스트 시스템 사양은 다음과 같습니다.
```
Ubuntu 22.04 LTS
Python 3.8.10 
Torch 1.9.0+cu111 
torchvision 0.10.0+cu111
CUDA 11.1
cuDnn 8.2.0    
```

❏ 사용 라이브러리 및 프로그램입니다.

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

# 실행 방법 (예시)

❏ 훈련 방법입니다.
```
python train_net.py  \
--config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml   \
--num-gpus 2 \
DATASETS.TRAIN multimodal_2022_train \
DATASETS.TEST multimodal_2022_val_with_sem_seg \
SOLVER.IMS_PER_BATCH 2 \
MODEL.MASK_FORMER.TEST.INSTANCE_ON False MODEL.MASK_FORMER.TEST.SEMANTIC_ON False  \
SOLVER.MAX_ITER 300000 \              # 원하는 학습량 설정
MODEL.SEM_SEG_HEAD.NUM_CLASSES 98  \  # category 개수에 따라 변경
OUTPUT_DIR ./<output_dir name>
```

❏ 평가 방법입니다.
```
python train_net.py  \
--config-file <output_dir name>/config.yaml
--num-gpus 2 \
--eval-only \
MODEL.WEIGHTS <output_dir name>/checkpoint_file \
DATASETS.EVAl <dataset_name>   # multimodal_2022_test_with_sem_seg
```



