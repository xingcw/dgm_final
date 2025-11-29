#! /bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

# cd PnPInversion || exit 1

MODEL_TYPE=sdxl

python run_editing_p2p.py \
    --output_path output/debug/${MODEL_TYPE}/default \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --edit_method_list directinversion+p2p \
    --data_path data \
    --model_type ${MODEL_TYPE}

# python evaluation/evaluate.py --metrics "structure_distance" \
#     "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
#     "ssim_unedit_part" "clip_similarity_source_image" \
#     "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
#     --result_path evaluation_result.csv \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --tgt_methods 1_directinversion+p2p

# python compare_methods.py \
#     --csv1 evaluation_result.csv --csv2 evaluation_result_original_p2p.csv \
#     --method1 "sd21_directinversion+p2p" --method2 "sd14_directinversion+p2p" \
#     --output comparison_results.txt
