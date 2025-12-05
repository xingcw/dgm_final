#! /bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

# cd PnPInversion || exit 1

# export MODEL_NAME=sd14
export MODEL_NAME=sd15

# python run_editing_controlnet.py \
#     --data_path data \
#     --output_path output/controlnet/sam \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --conditioning_scale 0.3 \
#     --controlnet_end_ratio 0.5 \
#     --use_sam

# python run_editing_controlnet.py \
#     --data_path data \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --output_path output/controlnet/ddim \
#     --conditioning_scale 0.3 \
#     --controlnet_end_ratio 1.0 \
#     --inversion_method ddim \
#     --use_sam  # or omit for Canny edges

# python run_editing_p2p.py \
#     --data_path data \
#     --output_path output/directinversion+p2p \
#     --edit_category_list 5 \
#     --edit_method_list directinversion+p2p

python evaluation/evaluate.py --metrics "structure_distance" \
    "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
    "ssim_unedit_part" "clip_similarity_source_image" \
    "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
    --result_path results/evaluation_result_sam+controlnet+p2p.csv \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --tgt_methods 1_directinversion+controlnet+p2p \
    --tgt_image_folders output/controlnet/sam/sam+controlnet+p2p/annotation_images

# python evaluation/evaluate.py --metrics "structure_distance" \
#     "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
#     "ssim_unedit_part" "clip_similarity_source_image" \
#     "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
#     --result_path results/evaluation_result_+p2p.csv \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --tgt_methods 1_ddim+p2p \
#     --tgt_image_folders output/ddim+p2p/annotation_images


python compare_methods.py \
    --csv1 results/evaluation_result_sd15_v1.csv --csv2 results/evaluation_result_sam+controlnet+p2p.csv \
    --method1 "directinversion+p2p" --method2 "directinversion+controlnet+p2p" \
    --output results/comparison_results_directinversion+p2p_vs_directinversion+controlnet+p2p.txt \
    --latex results/comparison_directinversion+p2p_vs_directinversion+controlnet+p2p.tex