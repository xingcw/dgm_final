#! /bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

# cd PnPInversion || exit 1

export MODEL_NAME=sd15_v1

python run_editing_p2p.py \
    --output_path output/debug/${MODEL_NAME} \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --edit_method_list directinversion+p2p \
    --data_path data

# python evaluation/evaluate.py --metrics "structure_distance" \
#     "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
#     "ssim_unedit_part" "clip_similarity_source_image" \
#     "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
#     --result_path results/evaluation_result_${MODEL_NAME}.csv \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --tgt_methods 1_directinversion+p2p \
#     --tgt_image_folders output/${MODEL_NAME}/directinversion+p2p/annotation_images

# python compare_methods.py \
#     --csv1 results/evaluation_result_ddimp2p.csv --csv2 results/evaluation_result_original_p2p.csv \
#     --method1 "ddim_p2p" --method2 "original_p2p" \
#     --output results/comparison_results_ddim_p2p.txt