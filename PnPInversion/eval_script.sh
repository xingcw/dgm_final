#! /bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

cd PnPInversion || exit 1

# export MODEL_NAME=sd14
export MODEL_NAME=sd15

# python run_editing_controlnet.py \
#     --data_path data \
#     --output_path output/${MODEL_NAME}/fallback_sam+controlnet+p2p \
#     --edit_category_list 0 \
#     --conditioning_scale 0.3 \
#     --controlnet_end_ratio 1.0 \
#     --use_sam

python run_editing_p2p.py \
    --data_path data \
    --output_path output/directinversion+p2p \
    --edit_category_list 0 \
    --edit_method_list directinversion+p2p

# python evaluation/evaluate.py --metrics "structure_distance" \
#     "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
#     "ssim_unedit_part" "clip_similarity_source_image" \
#     "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
#     --result_path results/evaluation_result_directinversion+p2p.csv \
#     --edit_category_list 5 \
#     --tgt_methods 1_directinversion+p2p \
#     --tgt_image_folders output/directinversion+p2p/annotation_images

# python compare_methods.py \
#     --csv1 results/evaluation_result_sd15_0.3sam.csv --csv2 results/evaluation_result_sd15_category5.csv \
#     --method1 "sam+controlnet+p2p" --method2 "directinversion+p2p" \
#     --output results/comparison_results_sam+controlnet+p2p_vs_directinversion+p2p.txt