export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

python src/visualize_mask_comparison.py \
    image_dir=data/plantsegv3/images/train \
    gt_dir=outputs/plantseg_binary/gt_binary_train \
    'mask_dirs=[{path: outputs/plantseg_binary/pseudo_masks, label: PSA+RW},{path: outputs/plantseg_binary/pseudo_masks_t_0.64, label: PSA+RW_t0.64},{path: outputs/plantseg_binary/weakclip_masks_fast, label: WeakCLIP}]' \
    output_dir=outputs/visualizations/pseudo_masks_t064_comparison \
    num_samples=20