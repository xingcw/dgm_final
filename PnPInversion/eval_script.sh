#! /bin/bash

cd PnPInversion || exit 1

python run_editing_p2p.py \
    --output_path output \
    --edit_category_list 0  \
    --edit_method_list directinversion+p2p \
    --data_path data


# python evaluation/evaluate.py --metrics "structure_distance" \
#     "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" \
#     "ssim_unedit_part" "clip_similarity_source_image" \
#     "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
#     --result_path evaluation_result.csv \
#     --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
#     --tgt_methods 1_directinversion+p2p