python train_net_video.py --num-gpus 1 --eval-only --config-file configs/lvvis/instance-segmentation/ov2seg_R50_bs16_50ep.yaml \
       	OUTPUT_DIR output/prompt3 MODEL.WEIGHTS output/prompt3/model_final.pth \
       	MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 MODEL.MASK_FORMER.DEC_LAYERS 7 MODEL.MASK_FORMER.CLIP_CLASSIFIER True \
       	DATASETS.TEST '("lvvis_val",)' MODEL.SEM_SEG_HEAD.NUM_CLASSES 1196 MODEL.MASK_FORMER.CLIP_PATH datasets/metadata/fg_bg_5_10_lvvis_ens.npy \
	INPUT.MIN_SIZE_TEST 480

       	#DATASETS.TEST '("ovis_val",)' MODEL.SEM_SEG_HEAD.NUM_CLASSES 25 MODEL.MASK_FORMER.CLIP_PATH datasets/metadata/fg_bg_5_10_ovis_ens.npy \
       	#DATASETS.TEST '("ytvis_2021_val",)' MODEL.SEM_SEG_HEAD.NUM_CLASSES 40 MODEL.MASK_FORMER.CLIP_PATH datasets/metadata/fg_bg_5_10_ytvis21_ens.npy \
       	#DATASETS.TEST '("ytvis_2019_val",)' MODEL.SEM_SEG_HEAD.NUM_CLASSES 40 MODEL.MASK_FORMER.CLIP_PATH datasets/metadata/fg_bg_5_10_ytvis19_ens.npy \
       	#DATASETS.TEST '("lvvis_test",)' MODEL.SEM_SEG_HEAD.NUM_CLASSES 1212 MODEL.MASK_FORMER.CLIP_PATH datasets/metadata/fg_bg_5_10_lvvis_ens.npy \
       	#DATASETS.TEST '("lvvis_val",)' MODEL.SEM_SEG_HEAD.NUM_CLASSES 1212 MODEL.MASK_FORMER.CLIP_PATH datasets/metadata/fg_bg_5_10_lvvis_ens.npy \


