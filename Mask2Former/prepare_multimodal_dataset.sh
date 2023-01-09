
DATA_PATH=$1
SPLIT_LIST=$2

IMG_DATAPATH=$DATA_PATH/"imgs/"
ANN_DATAPATH=$DATA_PATH/"labels/"

ANNOTATION_PATH=$DATA_PATH/"segmentation.json"
CATEGORY_PATH=$DATA_PATH/"category.json"

mkdir -p "datasets/multimodal/annotations"

for mode in train val test
do
	ln -s $IMG_DATAPATH "datasets/multimodal"/$mode
	ln -s $ANN_DATAPATH "datasets/multimodal/panoptic_"$mode
done

echo "MAKE DATASETS"
ln -s $CATEGORY_PATH "datasets/multimodal/annotations/categories.json"
python datasets/prepare_multimodal_panoptic_semantic.py \
	--annotation_path $ANNOTATION_PATH \
	--split_list $SPLIT_LIST \
	--category_json $CATEGORY_PATH

echo "COMPLETE!"
