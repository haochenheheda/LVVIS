python train_net.py --num-gpus 1 \
    --config-file configs/lvvis/instance-segmentation/ov2seg_R50_bs16_50ep_lvis.yaml \
    SOLVER.IMS_PER_BATCH 2 \
    MODEL.MASK_FORMER.CLIP_CLASSIFIER True \
    OUTPUT_DIR output/ov2seg \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
    MODEL.MASK_FORMER.DEC_LAYERS 7
