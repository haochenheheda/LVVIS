python train_net.py --num-gpus 1 --eval-only --config-file configs/lvvis/instance-segmentation/ov2seg_R50_bs16_50ep_lvis.yaml \
      	SOLVER.IMS_PER_BATCH 16 MODEL.MASK_FORMER.CLIP_CLASSIFIER True \
	OUTPUT_DIR output/prompt3 MODEL.WEIGHTS output/prompt3/model_final.pth \
	MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 MODEL.MASK_FORMER.DEC_LAYERS 7 \
	MODEL.SEM_SEG_HEAD.NUM_CLASSES 1196 \
	MODEL.MASK_FORMER.CLIP_PATH "datasets/metadata/fg_bg_5_10_lvvis_ens.npy" \
	DATASETS.TEST "('lvvis_oracle_val',)"


	#MODEL.SEM_SEG_HEAD.NUM_CLASSES 80 \
	#MODEL.MASK_FORMER.CLIP_PATH "datasets/metadata/fg_bg_5_10_coco_ens.npy" \
	#DATASETS.TEST "('coco_2017_val',)"
