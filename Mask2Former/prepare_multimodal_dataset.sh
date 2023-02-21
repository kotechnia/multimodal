#!/bin/bash

IMG_DATAPATH=$1
ANN_DATAPATH=$2
ANNOTATION_PATH=$3
CATEGORY_PATH=$4
SPLIT_LIST=$5

IMG_DATAPATH=$(realpath $IMG_DATAPATH)
ANN_DATAPATH=$(realpath $ANN_DATAPATH)
CATEGORY_PATH=$(realpath $CATEGORY_PATH)

mkdir -p "datasets/multimodal/annotations"

for mode in train val test
do
    if [ -L "datasets/multimodal"/$mode ] ; then
         rm "datasets/multimodal"/$mode
    fi
    ln -s $IMG_DATAPATH "datasets/multimodal"/$mode


    if [ -L "datasets/multimodal/panoptic_"$mode ] ; then
         rm "datasets/multimodal/panoptic_"$mode
    fi
    ln -s $ANN_DATAPATH "datasets/multimodal/panoptic_"$mode
done

echo "MAKE DATASETS"

if [ -L "datasets/multimodal/annotations/categories.json" ] ; then
    rm "datasets/multimodal/annotations/categories.json"
fi
ln -s $CATEGORY_PATH "datasets/multimodal/annotations/categories.json"

python datasets/prepare_multimodal_panoptic_semantic.py \
--annotation_path $ANNOTATION_PATH \
--split_list $SPLIT_LIST \
--category_json $CATEGORY_PATH

echo "COMPLETE!"
