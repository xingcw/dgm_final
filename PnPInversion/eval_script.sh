#! /bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

# cd PnPInversion || exit 1

# MODEL_TYPE="sd14_finetuned"

# python run_editing_p2p.py \
#     --output_path output/debug/${MODEL_TYPE} \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --edit_method_list directinversion+p2p \
#     --data_path data \
#     --model_type ${MODEL_TYPE}

# python evaluation/evaluate.py --metrics "structure_distance" \
#     "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
#     "ssim_unedit_part" "clip_similarity_source_image" \
#     "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
#     --result_path results/evaluation_result_${MODEL_TYPE}.csv \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --tgt_methods 1_directinversion+p2p \
#     --tgt_image_folders output/${MODEL_TYPE}/directinversion+p2p/annotation_images

python compare_methods.py \
    --csvs results/evaluation_result_sd14.csv results/evaluation_result_sd14_finetuned.csv results/evaluation_result_sdxl_finetuned.csv \
    --method-names "SD14" "SD14_tuned" "SDXL_tuned" \
    --output results/comparison_results_sd14_vs_sd14_finetuned_vs_sdxl_finetuned.txt \
    --latex results/comparison_results_sd14_vs_sd14_finetuned_vs_sdxl_finetuned.tex
