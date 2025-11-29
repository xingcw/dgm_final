#! /bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

# cd PnPInversion || exit 1

MODEL_TYPE=sdxl

python run_editing_p2p.py \
    --output_path output/${MODEL_TYPE} \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --edit_method_list directinversion+p2p \
    --data_path data \
    --model_type ${MODEL_TYPE}

python evaluation/evaluate.py --metrics "structure_distance" \
    "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
    "ssim_unedit_part" "clip_similarity_source_image" \
    "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
    --result_path results/evaluation_result_${MODEL_TYPE}.csv \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --tgt_methods 1_directinversion+p2p \
    --tgt_image_folders output/${MODEL_TYPE}/directinversion+p2p/annotation_images

python compare_methods.py \
    --csv1 results/evaluation_result_${MODEL_TYPE}.csv \
    --method1 "sdxl_directinversion+p2p" \
    --output results/comparison_results_${MODEL_TYPE}.txt
