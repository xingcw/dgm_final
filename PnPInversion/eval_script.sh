#! /bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd PnPInversion || exit 1

python run_editing_p2p.py \
    --output_path output/sd21 \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --edit_method_list directinversion+p2p \
    --data_path data \
    --model_type sd21

python evaluation/evaluate.py --metrics "structure_distance" \
    "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
    "ssim_unedit_part" "clip_similarity_source_image" \
    "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
    --result_path results/evaluation_result_sd21.csv \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --tgt_methods 1_directinversion+p2p \
    --tgt_image_folders output/sd21/directinversion+p2p/annotation_images

python compare_methods.py \
    --csvs results/evaluation_result_sd14.csv results/evaluation_result_sd15_v1.csv results/evaluation_result_sd21.csv  \
    --output results/comparison_results_sd14_vs_sd15_v1_vs_sd21_directinversion+p2p.txt \
    --latex results/comparison_results_sd14_vs_sd15_v1_vs_sd21_directinversion+p2p.tex
