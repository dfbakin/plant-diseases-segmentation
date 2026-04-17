# Plant Disease Segmentation: Research Context & Recap

> **Purpose**: Comprehensive reference for future agents / collaborators.
> **Author**: Denis Bakin — **Date**: March–April 2026
> **Repo**: `git@github.com:dfbakin/plant-diseases-segmentation.git`
> **Current branch**: `wsss-weakclip-pipeline`

---

## 1. Project Goal

Develop a **weakly-supervised semantic segmentation (WSSS)** pipeline that segments plant disease regions on leaf images using only **image-level labels** (no pixel-level annotations for training).

The research transfers established WSSS methods (MCTformer, PSA, WeakCLIP) from the Pascal VOC domain to the plant disease domain, using the **PlantSeg** dataset (pixel-level GT available for evaluation) and the **PlantVillage** dataset (image-level labels only).

A secondary goal is fully-supervised segmentation on PlantSeg, which serves as an upper-bound baseline.

---

## 2. Datasets

| Dataset | Role | #Classes | Labels | Size |
|---------|------|----------|--------|------|
| **PlantSegV3** (`data/plantsegv3/`) | Primary: WSSS pipeline & evaluation | 116 (1 bg + 115 diseases) | Pixel-level GT masks + image-level | ~7,700 images |
| **PlantVillage** (`data/plant-village/`) | Supplemental: image-level training | 38 folders (healthy + diseased) | Folder-based image-level | ~54,000 images |
| **Pascal VOC 2012** (`data/VOC2012/`) | Baseline: WSSS method validation | 21 (1 bg + 20 objects) | Pixel-level GT + image-level | ~10,000 images |
| **Plant Pathology 2020** (`data/plant-pathology-2020-fgvc7/`) | Auxiliary (not actively used) | 4 | Image-level | ~3,600 images |

**PlantVillage-to-PlantSeg mapping**: 115 PlantSeg disease classes are mapped from PlantVillage folder names via `src/data/plantvillage_mappings.py`. Healthy and PlantVillage-only classes are excluded from WSSS training.

All datasets are DVC-tracked (`data/*.dvc`).

---

## 3. Repository Structure

```
plant-diseases-segmentation/
├── src/                          # Python source (8,685 LOC)
│   ├── conf/                     # Hydra config dataclasses
│   │   ├── config.py             # Master SegmentationConfig
│   │   ├── augmentation.py       # Augmentation presets (spatial, color, etc.)
│   │   ├── classifier.py         # Classifier training config
│   │   ├── data.py               # DataConfig
│   │   ├── model.py              # ModelConfig
│   │   ├── spdnet.py             # SPDNet Hydra config (SPDNetConfig, data/model/trainer)
│   │   ├── trainer.py            # TrainerConfig
│   │   └── wsss.py               # WSSS-specific config
│   ├── data/                     # Dataset & datamodule
│   │   ├── plantseg.py           # PlantSeg constants: 115 disease names, color palette
│   │   ├── plantvillage.py       # PlantVillage dataset loader
│   │   ├── plantvillage_mappings.py  # PV folder -> PlantSeg disease ID mapping
│   │   ├── voc_classification.py # MCTformer dataset classes (VOC, PlantSeg binary,
│   │   │                         #   PlantSeg multiclass, PlantSeg+PV combined)
│   │   ├── voc_wsss.py           # VOC WSSS dataset
│   │   ├── datamodule.py         # Lightning DataModule
│   │   └── transforms.py         # Augmentation pipelines (albumentations)
│   ├── models/                   # Segmentation model zoo
│   │   ├── segformer.py          # SegFormer (MiT-B3 encoder)
│   │   ├── segnext/              # SegNeXt (MSCAN encoder + hamburger head)
│   │   ├── classification.py     # Classification models (timm-based)
│   │   ├── factory.py            # Model factory (dispatch by name)
│   │   └── base.py               # Lightning base module (train/val/test loops)
│   ├── metrics/                  # Evaluation metrics
│   │   ├── segmentation.py       # mIoU, per-class IoU, boundary IoU, dice
│   │   └── cam_evaluation.py     # CAM quality metrics (pointing acc, peak coverage)
│   ├── wsss/                     # WSSS pipeline modules
│   │   ├── mctformer/            # MCTformer-V2 classifier + CAM generator
│   │   │   ├── model.py          # DeiT-Small backbone + class/patch token interaction
│   │   │   ├── cam_generator.py  # Multi-scale CAM extraction from attention
│   │   │   └── evaluation.py     # CAM threshold sweep evaluation
│   │   ├── spdnet/               # SPDNet Siamese plant disease network (April 2026)
│   │   │   ├── model.py          # SPDNet: ResNet50 backbone + FPN + MSE + ADPLCam
│   │   │   │                     #   + SpatialCrossAttention (fusion_mode="spatial")
│   │   │   ├── dataset.py        # SiamesePlantSegDataset: pair sampling, multi-ref
│   │   │   ├── lightning.py      # SPDNetModule: Lightning training/validation
│   │   │   ├── cam_generator.py  # ADPL-CAM + feat_chmean/chvar/.../cam_classifier seed
│   │   │   │                     #   generation (accepts external ref_pool + class resolver)
│   │   │   └── class_resolver.py # Build same-class reference pool from train labels +
│   │   │                         #   resolve val image classes from filenames (REF BUG FIX)
│   │   ├── refinement/           # Mask refinement modules
│   │   │   ├── crf.py            # DenseCRF wrapper (la_crf + ha_crf)
│   │   │   ├── affinity_net.py   # PSA (Pixel Semantic Affinity) network
│   │   │   ├── random_walk.py    # Random Walk propagation
│   │   │   ├── aff_dataset.py    # Affinity label generation dataset
│   │   │   └── resnet38d.py      # ResNet-38d backbone for PSA
│   │   └── weakclip/             # WeakCLIP segmentation model
│   │       ├── model.py          # CLIP-based segmentation with text + visual branch
│   │       ├── clip_backbone.py  # Modified CLIP ViT-B/16 encoder
│   │       ├── clip_text_encoder.py  # Text encoder + context optimization
│   │       ├── context_decoder.py    # Cross-attention decoder
│   │       ├── decode_head.py    # FPN decode head with CRF loss
│   │       ├── fpn.py            # Feature Pyramid Network
│   │       ├── lightning.py      # Lightning training module
│   │       └── losses.py         # Seeding + boundary + identity losses
│   ├── training/                 # Shared training utilities
│   │   └── callbacks.py          # MLflow logging, checkpointing
│   ├── utils/
│   │   └── logging.py            # Rich logging setup
│   │
│   │  # ── Top-level scripts (Hydra entrypoints) ──
│   ├── train.py                  # Supervised segmentation training
│   ├── train_classifier.py       # timm classifier training (CAM benchmark)
│   ├── train_mctformer.py        # MCTformer-V2 training
│   ├── train_spdnet.py           # SPDNet Siamese training (Hydra entrypoint)
│   ├── train_psa.py              # PSA affinity net training
│   ├── train_weakclip.py         # WeakCLIP training
│   ├── train_deeplab_wsss.py     # DeepLabV3+ on pseudo-masks (VOC)
│   ├── evaluate.py               # Supervised eval
│   ├── evaluate_masks.py         # Mask-vs-GT evaluation (mIoU, per-class IoU)
│   ├── generate_cams.py          # CAM generation from MCTformer checkpoint
│   ├── generate_spdnet_cams.py   # ADPL-CAM generation from SPDNet checkpoint
│   ├── apply_crf.py              # DenseCRF post-processing of CAMs
│   ├── run_random_walk.py        # Random Walk pseudo-mask generation
│   ├── export_labels.py          # Image-level label export (npy dict)
│   ├── generate_weakclip_masks.py     # WeakCLIP inference (standalone)
│   ├── refine_weakclip_masks.py       # CRF refinement of WeakCLIP masks
│   ├── generate_refine_weakclip_masks.py  # Fused generate + refine (streaming)
│   ├── refine_masks_sam.py        # SAM1-based mask refinement
│   ├── analyze_mask_quality.py    # Comprehensive mask diagnostics
│   └── visualize_mask_comparison.py   # Side-by-side mask visualization
│
├── scripts/                       # Shell scripts (orchestration)
│   ├── run_plantseg_binary_pipeline.sh  # Main WSSS pipeline (10 steps)
│   ├── run_plantseg_pipeline.sh         # Multiclass pipeline (earlier)
│   ├── run_sam_refinement_experiments.sh # SAM1 experiment matrix (6 configs)
│   ├── smoke_binary_pipeline.sh         # CI smoke test
│   ├── train_mctformer_plantseg.sh      # MCTformer training (standalone)
│   ├── train_mctformer_voc.sh           # MCTformer training (VOC)
│   ├── train_weakclip.sh               # WeakCLIP training
│   ├── generate_cams.sh                # CAM generation
│   ├── apply_crf.sh                    # CRF application
│   ├── run_random_walk.sh              # Random Walk
│   ├── evaluate_masks.sh               # Mask evaluation
│   ├── export_labels.sh                # Label export
│   ├── train_psa.sh                    # PSA training
│   ├── generate_weakclip_masks.sh      # WeakCLIP inference
│   ├── refine_weakclip_masks.sh        # WeakCLIP CRF refinement
│   ├── evaluate_weakclip_masks.sh      # WeakCLIP evaluation
│   ├── visualize_pipeline_steps.sh     # Pipeline stage visualization
│   ├── train_deeplab_wsss.sh           # DeepLab WSSS training (VOC)
│   ├── compute_dataset_stats.py        # Dataset statistics
│   ├── visualize_predictions.py        # Prediction overlay visualization
│   ├── visualize_pseudo_masks.py       # Pseudo-mask visualization
│   ├── augment_ablation.sh             # Augmentation ablation script
│   ├── benchmark_models.sh             # Architecture benchmark script
│   ├── classifier_models_exp.sh        # Classifier-CAM benchmark script
│   ├── run_spdnet_experiments.sh       # SPDNet 6-run experiment sweep (N=1,3,5,8 × aug)
│   ├── run_spdnet_spatial_experiments.sh # Spatial cross-attention experiments (smoke+PS+PS_PV)
│   ├── visualize_spdnet_activations.py # SPDNet 8-panel activation grid visualizations
│   ├── visualize_spatial_attention.py  # Cross-attention map viz (head-avg, query-vs-ref grids)
│   ├── eval_spdnet_checkpoints.py      # Automated SPDNet checkpoint eval (CAM+sweep+viz)
│   ├── eval_spatial_runs.py            # Earlier eval driver (deprecated; precursor to _full)
│   ├── eval_spatial_full.py            # Full eval: feat_chmean/chvar + thr sweep +
│   │                                   #   per-distribution CRF tuning + viz (token+spatial)
│   ├── eval_spatial_overnight.sh       # Overnight wrapper for eval_spatial_full.py
│   ├── eval_cam_classifier.py          # cam_classifier (classifier projected on FUSED feats)
│   │                                   #   eval — only mode that uses spatial fusion output
│   ├── eval_cam_classifier_overnight.sh # Overnight wrapper for eval_cam_classifier.py
│   ├── smoke_test_ref_fix.py           # Smoke test: same-class refs picked correctly
│   ├── smoke_test_eval_pipeline.py     # Smoke test: feat_chmean/chvar pipeline end-to-end
│   ├── smoke_test_cam_classifier.py    # Smoke test: cam_classifier sensitive to fusion
│   └── overnight_eval_and_train.sh     # SPDNet overnight eval + training pipeline
│
├── tests/
│   ├── test_binary_pipeline.py    # Binary label/GT/CAM integration tests
│   ├── test_mctformer_model.py    # MCTformer forward-pass tests
│   ├── test_sam_refinement.py     # SAM refinement unit tests
│   └── test_spdnet.py            # SPDNet: 22 tests (forward, backward, grad flow,
│                                  #   multi-ref, reference sensitivity, CAM shapes)
│
├── reports/
│   ├── src/
│   │   ├── 01_datasets_ideas.qmd          # Dataset overview & ideas
│   │   ├── 02_architecture_benchmark.qmd  # SegFormer vs SegNeXt vs UNet etc.
│   │   ├── 03_classifier_cam_benchmark.qmd # Classifier-CAM quality report
│   │   ├── 05_mask_quality_analysis.qmd   # Binary vs MC115 pipeline comparison
│   │   └── 06_spdnet_spatial_findings.qmd # SPDNet spatial cross-attention findings
│   │                                       #   (architecture, seed extraction A/B/C, results)
│   ├── 06_spdnet_spatial_findings.pdf     # Rendered Beamer slides
│   ├── resources/                         # Generated figures & statistics (DVC)
│   └── _quarto.yml                        # Quarto project config
│
├── outputs/                       # Experiment outputs (DVC-tracked)
│   ├── plantseg_binary/           # Binary WSSS pipeline outputs
│   │   ├── labels/                # Image-level labels (.npy)
│   │   ├── gt_binary_train/       # Binary GT masks (train)
│   │   ├── gt_binary_val/         # Binary GT masks (val)
│   │   ├── cams/                  # CAMs + CRF outputs
│   │   ├── psa/                   # PSA checkpoint
│   │   ├── pseudo_masks_t_0.64/   # PSA+RW pseudo-masks (threshold 0.64)
│   │   ├── pseudo_masks_t_0.73/   # PSA+RW pseudo-masks (threshold 0.73)
│   │   └── weakclip_masks_t_0.64/ # WeakCLIP refined masks
│   ├── plantseg_binary_mc115/     # MC115 pipeline outputs (multiclass MCTformer)
│   │   ├── labels/                # 115-class + binary labels
│   │   ├── cams/                  # Aggregated binary CAMs (from 115 classes)
│   │   ├── psa/                   # PSA checkpoint
│   │   ├── pseudo_masks_t_0.73/   # PSA+RW pseudo-masks
│   │   └── weakclip_masks_t_0.73/ # WeakCLIP refined masks
│   ├── plantseg_wsss_26_cam_multiclass/ # Earlier multiclass CAMs (GT-masked, deprecated)
│   ├── spdnet_plantseg/           # SPDNet experiment outputs (DVC: spdnet_plantseg.dvc, ~34 GB)
│   │   ├── spdnet_fix_n1_heavy/   #   Token N=1 refs, heavy aug, 80 ep (best token CAMs)
│   │   │   └── checkpoints/       #     best.ckpt (ep69, mAP=0.859), last.ckpt
│   │   ├── spdnet_fix_n3_heavy/   #   N=3 refs, heavy aug, 57 ep (crashed, best classifier)
│   │   │   └── checkpoints/       #     best.ckpt (ep53, mAP=0.898), last.ckpt
│   │   ├── spdnet_fix_n3_light/   #   N=3 refs, light aug, 80 ep
│   │   │   └── checkpoints/       #     best.ckpt (ep74, mAP=0.894)
│   │   ├── spdnet_fix_n3_minimal/ #   N=3 refs, minimal aug, 80 ep (severe overfit)
│   │   │   └── checkpoints/       #     best.ckpt (ep79, mAP=0.821)
│   │   ├── spdnet_spatial_n1_ps/  #   Spatial cross-attn, PlantSeg-only, 80 ep (Apr 2026)
│   │   │   └── checkpoints/       #     best ep76 mAP=0.797, gate=0.333; last.ckpt
│   │   ├── spdnet_spatial_n1_ps_pv/ # Spatial cross-attn, PlantSeg+PV, 80 ep (Apr 2026)
│   │   │   └── checkpoints/       #     best ep76 mAP=0.888, gate=0.499; last.ckpt
│   │   ├── cams/                  #   Generated ADPL-CAMs (token, n1/n3 × max/top_energy)
│   │   ├── feature_seed_eval/     #   Initial 200-img feat_chmean+CRF sweep (Apr 2026)
│   │   ├── spdnet_token_n1_heavy_eval/      # Full-val (1247) eval, corrected refs:
│   │   │   ├── seeds_feat_chmean_corrected_refs/   #   1247 npy seed maps
│   │   │   ├── seeds_feat_chvar_corrected_refs/    #   1247 npy seed maps
│   │   │   ├── seeds_cam_classifier_max_corrected_refs/ # 1247 npy CAMs (most expensive!)
│   │   │   ├── crf_sweep_*.json                     #   CRF param sweep results
│   │   │   └── evaluation_results*.json             #   Final IoU metrics
│   │   ├── spdnet_spatial_n1_ps_eval/        # Same structure for spatial PS-only
│   │   ├── spdnet_spatial_n1_ps_pv_eval/     # Same structure for spatial PS+PV
│   │   ├── eval_summary_corrected_refs.json  # Aggregate summary (feat_chmean/chvar)
│   │   ├── eval_summary_cam_classifier.json  # Aggregate summary (cam_classifier)
│   │   └── spatial_eval_summary.json         # Earlier (buggy refs) summary, kept for record
│   ├── visualizations/            # Activation grid PNGs (DVC: visualizations.dvc, ~1.5 GB)
│   │   ├── val_cam_exploration/   #   MCTformer MC115 baseline (25 images)
│   │   ├── spdnet_val_cam_exploration/ # SPDNet initial run (broken model)
│   │   ├── spdnet_n1_best/        #   SPDNet N=1 best ckpt (25 images, 8-panel grids)
│   │   ├── spdnet_n3_best/        #   SPDNet N=3 best ckpt (25 images, 8-panel grids)
│   │   ├── spdnet_n3_last/        #   SPDNet N=3 last ckpt (25 images, 8-panel grids)
│   │   ├── feat_chmean_crf_comparison/ # First feat_chmean+CRF visualization batch
│   │   ├── spatial_attention_n1_ps{,_pv,_FIXED,_pv_FIXED}/ # Cross-attn maps (4 dirs)
│   │   ├── spdnet_token_n1_heavy_feat_chmean_crf_corrected_refs/  # Final viz batches:
│   │   ├── spdnet_token_n1_heavy_cam_classifier_max_crf_corrected_refs/
│   │   ├── spdnet_spatial_n1_ps_feat_chvar_crf_corrected_refs/
│   │   ├── spdnet_spatial_n1_ps_cam_classifier_max_crf_corrected_refs/
│   │   ├── spdnet_spatial_n1_ps_pv_feat_chvar_crf_corrected_refs/
│   │   └── spdnet_spatial_n1_ps_pv_cam_classifier_max_crf_corrected_refs/
│   ├── weakclip/                  # WeakCLIP checkpoints
│   └── weakclip_voc_pipeline/     # VOC WeakCLIP pipeline outputs
│
├── pretrained/                    # Pretrained weights (DVC)
│   ├── ViT-B-16.pt               # CLIP ViT-B/16 (for WeakCLIP)
│   └── res38_cls.pth             # ResNet-38d (for PSA backbone)
│
├── mlruns/                        # MLflow experiment tracking (DVC)
├── data/                          # Datasets (DVC-tracked)
├── pyproject.toml                 # Project metadata & dependencies
├── .cursor/rules/                 # Cursor agent rules (git/CI, behavior)
└── .dvc/                          # DVC configuration
```

---

## 4. WSSS Pipeline Architecture

The core WSSS pipeline (`scripts/run_plantseg_binary_pipeline.sh`) has 11 steps:

```
Image-Level Labels  ──>  MCTformer Classifier  ──>  CAMs
                              │
                              v
                     CRF Post-Processing (la_crf + ha_crf)
                              │
                              v
                     PSA Affinity Network Training
                              │
                              v
                     Random Walk Refinement ──> Pseudo-Masks
                              │
                              v
                     WeakCLIP Training (on pseudo-masks)
                              │
                              v
                     WeakCLIP Inference + CRF ──> Final Masks
```

### Pipeline Steps (Detail)

| Step | Script Entry | Module | Description |
|------|-------------|--------|-------------|
| 0a | `src/export_labels.py` | - | Export image-level labels (npy dict) for MCTformer training |
| 0b | `src/export_labels.py` | - | Export PlantSeg-only labels for CAM generation |
| 0c | inline Python | - | Convert multiclass GT masks to binary (bg=0, disease=1) |
| 1 | `src/train_mctformer.py` | `src/wsss/mctformer/model.py` | Train MCTformer-V2 image classifier |
| 2 | `src/generate_cams.py` | `src/wsss/mctformer/cam_generator.py` | Generate multi-scale CAMs from classifier attention |
| 3 | `src/apply_crf.py` | `src/wsss/refinement/crf.py` | Apply DenseCRF (low-alpha + high-alpha variants) |
| 4 | `src/evaluate_masks.py` | `src/metrics/segmentation.py` | Evaluate CRF masks vs GT |
| 5 | `src/train_psa.py` | `src/wsss/refinement/affinity_net.py` | Train Pixel Semantic Affinity network |
| 6 | `src/run_random_walk.py` | `src/wsss/refinement/random_walk.py` | Random Walk propagation -> pseudo-masks |
| 7 | `src/evaluate_masks.py` | - | Evaluate pseudo-masks vs GT |
| 8 | `src/train_weakclip.py` | `src/wsss/weakclip/lightning.py` | Train WeakCLIP on pseudo-masks |
| 9 | `src/generate_refine_weakclip_masks.py` | `src/wsss/weakclip/model.py` | Generate + CRF-refine WeakCLIP masks (streaming) |
| 10 | `src/evaluate_masks.py` | - | Evaluate final WeakCLIP masks vs GT |

### Pipeline Variants

The pipeline supports two MCTformer modes via environment variables:

**Binary mode** (default):
```bash
./scripts/run_plantseg_binary_pipeline.sh
```
- `MCTFORMER_DATASET=plantseg_binary` (1 fg class: "disease")
- MCTformer classifies "has disease" vs "no disease"

**MC115 mode** (multiclass-to-binary):
```bash
MCTFORMER_DATASET=plantseg_with_pv \
MCTFORMER_EXPERIMENT=mctformer_plantseg_mc115_pv \
BINARY_AGGREGATE=max \
OUT_BASE=outputs/plantseg_binary_mc115 \
BINARY_BASE=outputs/plantseg_binary \
WEAKCLIP_EXPERIMENT=weakclip-plantseg-binary-mc115-t_0.73 \
scripts/run_plantseg_binary_pipeline.sh
```
- `MCTFORMER_DATASET=plantseg_with_pv` (115 fg classes: specific diseases)
- MCTformer classifies each disease individually -> 115-channel CAMs
- `BINARY_AGGREGATE=max` takes `np.max` across all 115 class channels -> single binary CAM
- Downstream pipeline (CRF, PSA, RW, WeakCLIP) operates in 2-class mode
- `BINARY_BASE=outputs/plantseg_binary` reuses GT masks and binary labels from binary run

**Key environment variables**:
- `SKIP_STEPS="0,1,2"` — skip specific steps
- `MCTFORMER_CKPT=path/to/last.ckpt` — reuse existing checkpoint
- `WEAKCLIP_QUALITY=fast|full` — fast: single-scale; full: 5 scales + flip
- `OUT_BASE` — output directory
- `BINARY_BASE` — directory for shared binary resources (GT, labels)

---

## 5. Experiments Conducted

### 5.1 Fully-Supervised Segmentation (Upper Bound)

#### 5.1.1 Architecture Benchmark
**Experiment**: `plantseg_architecture_benchmark` (MLflow ID: `358457224855191874`)
**Goal**: Compare segmentation architectures on PlantSeg (binary, 2-class).

| Architecture | mIoU | Disease IoU | BG IoU | Notes |
|-------------|------|-------------|--------|-------|
| DeepLabV3+ (ResNet-50) | 78.9% | 64.6% | 92.9% | Baseline |
| U-Net (ResNet-34) | 78.7% | 64.6% | 92.8% | Dice loss |
| SegFormer (MiT-B3) | 80.9% | 68.2% | 93.6% | Best among small models |
| **SegNeXt (MSCAN-T)** | **82.1%** | **70.1%** | **94.0%** | **Best overall** |

**Config**: image_size=384, lr=0.0002-0.0003, 30 epochs, AdamW, cross-entropy loss.

#### 5.1.2 Augmentation Ablation
**Experiment**: `plantseg_augmentation_ablation_fp32_final` (MLflow ID: `591324269892947917`)
**Goal**: Systematic comparison of augmentation strategies (13 runs).

Both SegFormer and SegNeXt tested with augmentation presets:
- `baseline` (no augmentation)
- `spatial_light` / `spatial_heavy` (flips, rotations, crops)
- `color_natural` / `artificial_color` (brightness, contrast, hue)
- `spatial_color_light` (combined)
- `noise_blur` (Gaussian noise, blur)
- `full` (everything)

**Key findings**:
- `spatial_color_light` achieved best SegFormer mIoU (81.0%) with good generalization
- Heavy augmentation caused overfitting gap reduction but did not always improve test mIoU
- SegNeXt baseline was strong (80.2%) even without augmentation, suggesting architectural robustness
- Best overall: **SegNeXt + full augmentation** = 81.5% test mIoU

#### 5.1.3 Multiclass Segmentation
**Experiment**: `plantseg_multiclass_benchmark` (MLflow ID: `319358957156901464`)
**Goal**: 116-class segmentation on PlantSeg.

| Model | mIoU | Dice | Notes |
|-------|------|------|-------|
| SegNeXt (spatial_color_light) | 44.2% | 57.8% | Difficult task with 116 classes |

The dramatic drop from binary (82.1%) to multiclass (44.2%) motivated the WSSS binary-framing approach.

### 5.2 Classifier-CAM Quality Benchmark

#### 5.2.1 Traditional Classifiers (GradCAM)
**Experiment**: `dfbakin_classifier_cam_benchmark` (MLflow ID: `675534359741067840`)
**Goal**: Evaluate how well standard classifiers localize diseases via GradCAM.

| Classifier | Accuracy | CAM IoU | CAM F1 | Pointing Acc |
|-----------|----------|---------|--------|--------------|
| ResNet-18 | 68.5% | 20.5% | 30.8% | 46.7% |
| ResNet-50 | 72.7% | 15.2% | 24.2% | 45.9% |
| EfficientNet-B0 | 74.2% | 20.6% | 30.8% | 45.8% |

**Config**: 120 classes (PlantSeg), GradCAM, threshold=0.5, 30 epochs.
**Conclusion**: Standard GradCAM produces poor localization. MCTformer's attention-based approach is significantly better.

### 5.3 WSSS Pipeline (VOC Validation)

The pipeline was first validated on Pascal VOC to ensure correctness before applying to PlantSeg.

#### 5.3.1 MCTformer on VOC
**Experiment**: `mctformer_voc_v2` (MLflow ID: `229846199444711023`)
- 21-class MCTformer-V2 on VOC2012
- Used to generate CAMs -> CRF -> PSA -> RW -> pseudo-masks

#### 5.3.2 WeakCLIP on VOC
**Experiment**: `weakclip-voc` (MLflow ID: `301684071360501862`)
- Best val mIoU: **64.0%** (4 runs with different learning rates)
- Validated the full pipeline end-to-end on a standard benchmark

#### 5.3.3 DeepLab WSSS on VOC
**Experiment**: `deeplab_wsss_voc` (MLflow ID: `826207548213372587`)
- DeepLabV3+ (ResNet-101) trained on VOC pseudo-masks
- val mIoU: 51.4% (reasonable for WSSS)

### 5.4 WSSS Pipeline on PlantSeg

#### 5.4.1 Binary Pipeline (Baseline)
**Output directory**: `outputs/plantseg_binary/`
**MLflow experiments**:
- `mctformer_plantseg_binary` (ID: `592872031317447660`) — binary MCTformer (1 fg class)
  - val mAP: 99.99% (trivial binary task)
- `weakclip-plantseg-binary` (ID: `206029539984518366`) — WeakCLIP on binary pseudo-masks
  - val mIoU: 45.0% (threshold 0.64)
- `weakclip-plantseg-binary-t_0.64` (ID: `444840611733456632`) — threshold 0.64 variant
  - val mIoU: 47.1%
- `weakclip-plantseg-binary-t_0.73` (ID: `113502554219633766`) — threshold 0.73 variant
  - val mIoU: 46.5%

**Pipeline evaluation results (2-class, binary GT)**:

| Stage | mIoU | BG IoU | Disease IoU |
|-------|------|--------|-------------|
| LA-CRF | 45.01% | 77.37% | 12.64% |
| HA-CRF | 45.44% | 76.52% | 14.36% |
| PSA+RW | 43.01% | 65.80% | 20.21% |
| WeakCLIP | 46.22% | 66.93% | 25.52% |

**Key observation**: Binary MCTformer achieves near-perfect classification accuracy (mAP ~100%), suggesting it learns a trivial signal. The resulting CAMs are diffuse and poorly localized, leading to low disease IoU throughout the pipeline.

#### 5.4.2 Multiclass MCTformer on PlantSeg (Historical)
**Experiment**: `mctformer_plantseg` (MLflow ID: `771976166740944756`)
- 115-class MCTformer on PlantSeg only (no PlantVillage augmentation)
- val mAP: 76.9% (much harder task forces better localization)
- Generated CAMs were stored in `outputs/plantseg_wsss_26_cam_multiclass/`

**Critical finding**: These historical CAMs were **GT-masked** during generation — `src/generate_cams.py` line 114 applied `cls_att * target.view(...)`, zeroing out all non-GT class activations. This made the "multiclass" CAMs effectively single-class, discarding cross-disease activation signals. This discovery motivated full regeneration.

#### 5.4.3 MC115 Pipeline (Multiclass-to-Binary)
**Output directory**: `outputs/plantseg_binary_mc115/`
**Hypothesis**: Training MCTformer on 115 specific diseases forces it to learn more precisely localized activations (distinguishing e.g., "apple scab" from "apple rust" requires attending to specific lesion patterns). These more focused CAMs, when max-aggregated into a single disease channel, produce a better binary signal than the diffuse binary CAMs.

**Implementation changes**:
1. `src/data/voc_classification.py`: Added `PlantSegMCTformerDataset` support for `plantseg_with_pv` mode combining PlantSeg + PlantVillage training data with 115-class labels
2. `src/export_labels.py`: Added `plantseg_wsss_with_pv` label export mode (115 classes, PlantSeg + PV)
3. `src/generate_cams.py`: Added `binary_aggregate` parameter; when set to `"max"`, removes GT-masking and takes `np.max` across all 115 class CAMs -> single binary CAM
4. `scripts/run_plantseg_binary_pipeline.sh`: Made fully configurable for both binary and MC115 modes

**MC115 pipeline results (2-class, binary GT)**:

| Stage | mIoU | BG IoU | Disease IoU |
|-------|------|--------|-------------|
| LA-CRF | 51.94% | 76.10% | 27.77% |
| HA-CRF | 53.49% | 75.81% | 31.16% |
| PSA+RW | 42.29% | 54.17% | 30.42% |
| WeakCLIP | 46.36% | 59.40% | 33.31% |

**MC115 vs Binary comparison**:

| Stage | Binary Disease IoU | MC115 Disease IoU | Relative Improvement |
|-------|-------------------|-------------------|---------------------|
| LA-CRF | 12.64% | 27.77% | +120% |
| HA-CRF | 14.36% | 31.16% | +117% |
| PSA+RW | 20.21% | 30.42% | +50% |
| WeakCLIP | 25.52% | 33.31% | +31% |

**Hypothesis confirmed**: MC115 dramatically improves disease detection at every stage. The key trade-off is reduced BG IoU (more false positives), indicating increased recall at the cost of precision. Overall mIoU is comparable or better.

### 5.5 CAM Threshold Sweep: mIoU vs Disease IoU

**Module**: `src/wsss/mctformer/evaluation.py`

The original `evaluate_cam_threshold_sweep` optimized for overall mIoU, which is dominated by the background class in binary segmentation. Two bugs were fixed:
1. **Wrong optimization target**: mIoU averages BG and disease IoU; for binary tasks, BG dominates pixels, biasing the threshold toward high values that sacrifice disease recall.
2. **Premature early stopping**: The original code broke on the first mIoU decrease, missing the global optimum.

**Fix**: Added `optimize_metric` parameter (`"mIoU"`, `"disease_iou"`, or any class name), removed the early `break`, added optional `patience` parameter, and returned full per-threshold curves.

**Trial run on MC115 CAMs** (500-sample subset):

| Optimize for | Threshold | BG IoU | Disease IoU | mIoU |
|---|---|---|---|---|
| mIoU (legacy) | 0.73 | 76.26% | 27.40% | 51.83% |
| Disease IoU | 0.59 | 68.10% | 29.98% | 49.04% |

Disease-optimized threshold recovers +2.58pp disease IoU (+9.4% relative) by accepting more disease pixels. The mIoU metric is a poor proxy for disease detection in binary segmentation.

### 5.6 SAM1 Mask Refinement

**Script**: `scripts/run_sam_refinement_experiments.sh` (v3, MC115 pipeline)
**Module**: `src/refine_masks_sam.py`
**Model**: `facebook/sam-vit-huge`

**Motivation**: Post-process WSSS pseudo-masks with SAM1's boundary-precise segmentation.

#### 5.6.1 Original Experiments (v1, Binary Pipeline)

**Experiment matrix (6 configurations, A-F)** using binary pipeline outputs:

| ID | Prompt Mode | Mask Selection | Input Mask |
|----|-------------|----------------|------------|
| A | mask_only | best_iou | PSA+RW |
| B | mask_only | smallest_area | PSA+RW |
| C | box_only | best_iou | PSA+RW |
| D | box_only | smallest_area | PSA+RW |
| E | mask_and_points | best_iou | PSA+RW |
| F | box_and_points | smallest_area | PSA+RW |

**Results**: Massive oversegmentation (~2% disease IoU). SAM1 tends to segment the entire leaf when given diffuse prompts.

#### 5.6.2 MC115 Experiments (v3)

New experiment matrix using MC115 pipeline outputs, with additional prompt modes:

**Prompt modes**:
- `mask_only`: feed pseudomask as dense logit prompt (binary ±6.0 logits)
- `soft_mask`: continuous probability map → graded logits (bilinear resize, configurable `logit_scale`)
- `soft_mask_and_points`: soft mask + CAM-derived positive/negative points
- `box_only`: compute bounding box from mask; SAM segments freely inside
- `box_and_points`: bounding box + CAM-derived anchor points

**New `prob_to_logits` function**: Converts continuous [0,1] probability maps to graded SAM logits (vs binary mask_to_logits which saturates at ±6.0). Uses bilinear interpolation and optional `confidence_threshold` gating.

**Mask selection**: All v3 experiments use `smallest_area` to counter oversegmentation.

| ID | Input | Prompt Mode | Disease IoU | BG IoU | mIoU |
|----|-------|-------------|-------------|--------|------|
| G | MC115 WeakCLIP | mask_only | **0.73%** | 79.56% | 40.14% |
| H | MC115 CAM (soft, s=3.0) | soft_mask | **0.51%** | 79.86% | 40.18% |
| I | MC115 CAM (gated p>0.3) | soft_mask | **0.51%** | 79.91% | 40.21% |
| J | WeakCLIP bbox + CAM pts | box_and_points | **33.96%** | 74.32% | 54.14% |
| K | CAM soft + CAM pts | soft_mask_and_points | **17.39%** | 75.65% | 46.52% |
| J_tuned | WeakCLIP bbox + 20 pts (q=0.99) | box_and_points | **23.79%** | 77.75% | 50.77% |

**Critical finding — prompt type determines success or failure:**

- **Dense mask prompts (G, H, I) catastrophically fail** (<1% disease IoU). SAM interprets any dense mask as "the region of interest" and snaps to the nearest object boundary — the leaf edge. Disease information is destroyed.
- **Geometric prompts (J) work**: bbox from WeakCLIP constrains SAM's spatial attention; CAM-derived positive/negative points give explicit "this IS / is NOT disease" anchors. SAM uses its image features to find texture-coherent regions within these constraints.
- **Soft mask + points (K)**: The dense soft mask signal partially overrides the point anchors, degrading performance compared to pure geometric prompts.
- **J vs WeakCLIP input**: SAM-J achieves +0.65pp disease IoU and **+14.92pp BG IoU** over its WeakCLIP input, with +7.78pp overall mIoU. SAM acts as a precision filter — it doesn't find new disease, but removes false positive background pixels.

#### 5.6.3 Point Sampling Quality Analysis

**Script**: `scripts/analyze_point_sampling_quality.py`

Diagnostic analysis of how well CAM-derived points align with ground truth (300-image subsample).

**Positive point precision vs quantile** (num_pos=5, neg_q=0.05):

| Pos quantile | Pos precision | Avg CAM value |
|---|---|---|
| 0.70 | 34.7% | 0.668 |
| 0.80 | 40.5% | 0.727 |
| 0.90 | 45.5% | 0.813 |
| 0.95 | 48.9% | 0.872 |
| 0.99 | 53.3% | 0.948 |

**Negative point precision**: >94% at all settings. Background CAM signal is clean.

**Key findings**:
- Even at the 99th percentile, only ~53% of positive points land on GT disease. The CAM's high-activation peaks often fall on healthy tissue.
- Point count does not affect precision (farthest-point sampling maintains spatial diversity). The ceiling is set by CAM localization quality.
- Increasing from q=0.90 (J's setting, 45.5% precision) to q=0.99 (53.3%) is beneficial but with diminishing returns.

#### 5.6.4 Point Count Tuning (J_tuned)

Experiment J was re-run with q=0.99 and 20 positive / 15 negative points:
- Disease IoU: 23.79% (vs 33.96% for original J with q=0.90, 5/5 points)
- BG IoU: 77.75% (vs 74.32%)

**Counterintuitive result**: More aggressive point settings (higher quantile, more points) produced *worse* disease IoU but *better* BG IoU. This suggests that q=0.99 positive points, while more precise individually, are too concentrated in small high-activation peaks, causing SAM to produce very tight segments that miss diffuse disease regions. The original q=0.90 with 5 points provides a better spatial coverage vs precision trade-off.

**SAM1 text prompts**: Investigation of the SAM1 paper confirmed that while the architecture supports text prompts, the publicly released checkpoints only accept geometric prompts (points, boxes, masks). Text-prompted segmentation would require specialized training (e.g., GroundedSAM). Deferred to future work.

### 5.7 SPDNet Siamese Network Experiments (April 2026)

**Motivation**: MCTformer CAMs (both binary and MC115) were visualized on 25 validation images with multiple activation types (binary-agg CAM, per-class top-1/top-2 attention, GradCAM, CLS/patch token attention). **Conclusion**: MCTformer does not discriminate distinctive disease features — hot spots occasionally coincide with disease spots but this is not reliably confirmed across samples. This motivated implementing the Siamese approach from the paper (see Section 13 for full details).

**MLflow experiment**: `spdnet_plantseg` (ID: `285465004951754042`)
**Output directory**: `outputs/spdnet_plantseg/` (DVC-tracked: `outputs/spdnet_plantseg.dvc`)

#### 5.7.1 SPDNet Architecture (Implemented)

Siamese network with shared-weight ResNet50 backbone processing (query, reference) image pairs:

```
Query Image ─┐                          Reference Image(s) ─┐
             ├─ ResNet50 (shared) ──┐                        ├─ ResNet50 (shared) ──┐
             │                      │                        │                      │
             v                      v                        v                      v
        layer1..layer4         layer1..layer4           layer1..layer4         layer1..layer4
             │                      │                        │                      │
             v                      v                        v                      v
         FPN (4 levels, 256ch)  MSE (channel attn)       FPN (4 levels, 256ch)  MSE (channel attn)
             │                                               │
             v                                               v
         query FPN features                             ref FPN features
             │                                               │
             └──────────── ADPL-CAM Token Fusion ────────────┘
                                    │
                                    v
                          Fused features → GAP → Linear(256, 115) → logits
```

**Key components**:
- **Backbone**: `torchvision.models.resnet50(pretrained=True)`, 4 stage outputs (256, 512, 1024, 2048 channels)
- **FPN**: 4 lateral connections + top-down pathway, all outputs 256 channels. Each level applies `Conv2d(in, 256, 1)` lateral + `Conv2d(256, 256, 3, padding=1)` smooth
- **MSE (Multi-Scale Excitation)**: Channel attention per FPN level. `GAP(F) + GMP(F) → FC(256→64) → ReLU → FC(64→256) → Sigmoid → F * attention`. Applied **symmetrically** to both query AND reference FPN features
- **ADPL-CAM token fusion**: Per FPN level, `GlobalMaxPool(ref_features) → token T_i` (shape `[B, 256]`). Fuse: `fused = query + α * Σ T_i.unsqueeze(-1).unsqueeze(-1)`, where `α` is a learnable scalar (init 0.1). For N>1 references, tokens are **averaged** across references before fusion
- **Classifier**: `AdaptiveAvgPool2d(1) → flatten → Linear(256, num_classes=115)`
- **Loss**: `MultiLabelSoftMarginLoss` (multi-label classification, not contrastive)
- **Optimizer**: AdamW, lr=1e-4 base (scaled by batch_size/32), weight_decay=0.05
- **Scheduler**: CosineAnnealingLR with 5-epoch linear warmup

**Multi-reference support**: At training time, N same-class references are sampled per query. Each reference is processed through the full backbone+FPN+MSE pipeline **with gradients** (no `torch.no_grad()`). Tokens from all N references are averaged per FPN level before fusion. This increases memory/compute linearly with N.

**Data pipeline**:
- `SiamesePlantSegDataset` wraps the existing PlantSeg+PlantVillage combined dataset
- `_build_index()` creates a `Dict[int, List[int]]` mapping class_id → sample indices by reading mask files directly (avoids loading images)
- `__getitem__` returns `{"image": query, "ref_images": [ref1, ..., refN], "label": multi_hot_115}`
- `siamese_collate_fn` stacks each reference position into `[B, 3, H, W]` tensors

#### 5.7.2 Critical Implementation Bug Found & Fixed

**Bug (initial implementation)**: `SPDNet.forward()` only invoked ADPL-CAM token fusion when `return_cam=True` (inference mode). During training (`return_cam=False`), reference features were computed but **completely discarded** — classification used only raw query features. Consequences:
1. The model trained as a **plain ResNet50+FPN classifier** — the Siamese structure was inert
2. `adpl_cam.alpha` (learnable fusion weight) received **zero gradient** — stuck at init value 0.1
3. Reference images were wasted compute during training (backbone ran on them for nothing)
4. At inference, fusion ran with untrained alpha and untrained token interaction

**How it was discovered**: Unit test `test_backward_pass` explicitly checked every named parameter for `grad is not None`. `adpl_cam.alpha` was the only parameter with zero gradient, flagging the issue.

**Additional bugs fixed simultaneously**:
- **Asymmetric MSE**: Channel attention was applied to query FPN features only, not reference features. Fixed to apply `self.mse` to both, so reference tokens carry channel-attended information
- **Unnormalized query_merged**: The sum of 4 FPN levels was used directly without normalization (`query_merged = sum(q_fpn)`). When fusing with reference tokens of magnitude ~1, the feature scale mismatch was ~4×. Fixed: `query_merged = sum(q_fpn) / len(q_fpn)`
- **Dead `channels` parameter**: `ADPLCam.__init__` accepted an unused `channels` arg. Removed.
- **Slow dataset startup**: `_build_index()` loaded every image through the full transform pipeline just to read labels. Fixed to read labels directly from mask `.png` files and `_pv_samples` metadata

**Verification**:
- `test_backward_pass`: All parameters (including `adpl_cam.alpha`) now receive non-zero gradients ✓
- `test_reference_sensitivity`: Logits change when different references are used for same query ✓
- `test_multi_reference_forward`: Model handles list of N reference tensors, output shapes correct ✓
- `test_multi_reference_backward`: All parameters receive gradients in multi-reference mode ✓
- 3-epoch smoke run: val/mAP rose from random to ~0.3 in 3 epochs (vs stuck at ~0.15 with broken model) ✓

#### 5.7.3 SPDNet Classification Results

| Run Name | N refs | Augmentation | Epochs | Best val/mAP | Final train/mAP | Train-Val Gap | Notes |
|----------|--------|-------------|--------|-------------|-----------------|--------------|-------|
| `spdnet_448_42` (BROKEN) | 1 | heavy | 45 | 0.621 | 0.476 | — | Reference ignored (bug), baseline |
| `spdnet_fix_n1_heavy` | 1 | heavy | 80 | **0.859** | 0.889 | 3.0pp | Mild overfit, best CAMs |
| `spdnet_fix_n3_heavy` | 3 | heavy | 57* | **0.898** | 0.965 | 6.7pp | Best classifier, moderate overfit |
| `spdnet_fix_n3_light` | 3 | light | 80 | **0.894** | 0.990 | 9.6pp | Significant overfit |
| `spdnet_fix_n3_minimal` | 3 | minimal | 80 | 0.821 | 0.999 | 17.8pp | Severe overfit |
| `spdnet_fix_n5_heavy` | 5 | heavy | — | — | — | — | Not run (est. ~5.2h) |
| `spdnet_fix_n8_heavy` | 8 | heavy | — | — | — | — | Not run (est. ~7.8h) |

*`spdnet_fix_n3_heavy` crashed at epoch 57 (OOM during checkpoint save), but had already plateaued. Best checkpoint at epoch 53.

**Augmentation configurations**:
- **heavy**: `timm.create_transform` with RandAugment (n=2, m=9), ColorJitter(0.4,0.4,0.4), RandomErasing(0.25), RandomResizedCrop(448)
- **light**: `RandomResizedCrop(448)` + `RandomHorizontalFlip` + `ColorJitter(0.3,0.3,0.3,0.1)` + `Normalize`
- **minimal**: `Resize(512)` + `CenterCrop(448)` + `RandomHorizontalFlip` + `Normalize`
- **val** (all runs): `Resize(512)` + `CenterCrop(448)` + `Normalize`

**Training hyperparameters** (all runs):
- Image size: 448×448, batch_size: 16 (N=1) or 12 (N=3), gradient_accumulation: 2 (N=3)
- Effective batch size: 32 (N=1) or 24 (N=3)
- lr: 1e-4 × (eff_batch/32), weight_decay: 0.05, warmup: 5 epochs
- `log_every_n_steps=200`, `ModelCheckpoint` monitoring `val/mAP` (mode=max)
- Mixed precision: bf16-mixed

**Key findings (classification)**:
1. **Architecture fix was transformative**: broken 0.621 → fixed 0.859 = **+38pp absolute**. Confirms the Siamese fusion was completely inactive before
2. **Multi-reference helps classification**: N=3 (0.898) > N=1 (0.859) by +3.9pp. More same-class context improves predictions
3. **Heavy augmentation is critical**: minimal aug → severe overfitting (train 0.999 vs val 0.821, 17.8pp gap); heavy aug gave best val/train balance (3.0pp gap for N=1, 6.7pp for N=3)
4. **Light aug ≈ heavy aug for N=3**: 0.894 vs 0.898 — within noise — but light showed more overfitting (9.6pp gap)
5. **Overfitting is the dominant problem**: every run except N=1 heavy shows train mAP approaching 1.0 while val plateaus

#### 5.7.4 SPDNet CAM Quality (Disease Localization)

CAMs generated on full PlantSeg val set (1247 images) using multi-scale inference + horizontal flip. Two binary aggregation modes tested:
- **max**: `np.max` across all 115 per-class ADPL-CAMs → single binary CAM (same as MC115 MCTformer)
- **top_energy**: use only the single class with highest total CAM energy sum

Full threshold sweep (0.01–0.99 step 0.01) optimizing for disease IoU:

| Checkpoint | Epoch | N | Agg Mode | Best Thresh | Disease IoU | BG IoU | mIoU |
|-----------|-------|---|----------|------------|-------------|--------|------|
| **n1_best** | 69 | 1 | **max** | 0.23 | **28.08%** | 73.09% | **50.59%** |
| n1_best | 69 | 1 | top_energy | 0.23 | 27.32% | 72.75% | 50.03% |
| n3_best | 53 | 3 | max | 0.16 | 24.61% | 64.30% | 44.45% |
| n3_best | 53 | 3 | top_energy | 0.12 | 23.72% | 55.96% | 39.84% |
| n3_last | 56 | 3 | max | 0.16 | 24.40% | 65.83% | 45.11% |
| n3_last | 56 | 3 | top_energy | 0.14 | 23.26% | 60.44% | 41.85% |

**Comparison with MCTformer MC115** (the method SPDNet was intended to improve upon):

| Method | Disease IoU | BG IoU | mIoU | Threshold | Notes |
|--------|------------|--------|------|-----------|-------|
| MC115 MCTformer (HA-CRF) | **31.16%** | 75.81% | **53.49%** | 0.73 | After CRF post-processing |
| MC115 MCTformer (raw CAM) | **~29.98%** | 68.10% | 49.04% | 0.59 | Disease-IoU optimized, 500-sample |
| SPDNet n1_best (raw CAM) | 28.08% | 73.09% | 50.59% | 0.23 | Best SPDNet config |
| SPDNet n3_best (raw CAM) | 24.61% | 64.30% | 44.45% | 0.16 | Better classifier, worse CAMs |

**Key findings (CAM quality)**:
1. **SPDNet CAMs are comparable but slightly below MC115 MCTformer**: best SPDNet disease IoU is 28.08% vs MCTformer's ~29.98% (on raw CAMs without CRF). The gap would widen after CRF post-processing (MCTformer HA-CRF reaches 31.16%)
2. **N=1 produces better CAMs than N=3** despite N=3 having +3.9pp higher classification mAP (0.898 vs 0.859). Hypothesis: the multi-reference model overfits to the reference signal, producing more confident but less spatially accurate activations
3. **Max aggregation consistently outperforms top_energy** across all checkpoints (0.5–3.5pp disease IoU better). Max captures activation from multiple disease classes; top_energy misses secondary signals
4. **SPDNet needs much lower thresholds** (0.12–0.23 vs 0.59–0.73 for MCTformer): ADPL-CAMs have lower contrast and more diffuse activation patterns, requiring aggressive binarization
5. **The Siamese reference-guidance did NOT achieve the hoped-for localization improvement** over the simpler MCTformer attention-based approach on PlantSeg. The paper's impressive visual results appear to be cherry-picked from a tiny 42-image dataset

#### 5.7.5 SPDNet Visualization Details

8-panel grids generated per validation image (`scripts/visualize_spdnet_activations.py`):

| Panel | Content | Purpose |
|-------|---------|---------|
| 1 | Original image + GT mask overlay (green) | Ground truth reference |
| 2 | Binary-aggregated CAM (max across 115 classes) | Primary localization output |
| 3 | Top-1 class ADPL-CAM | Strongest per-class activation |
| 4 | Top-2 class ADPL-CAM | Secondary disease signal |
| 5 | Query features (before fusion) | What backbone sees without reference |
| 6 | Fused features (after reference token injection) | What classification head sees |
| 7 | Reference contribution: `fused - query` | Isolated effect of Siamese branch |
| 8 | GradCAM (from classifier gradients → fused features) | Standard gradient-weighted CAM |

Generated for 25 val images per checkpoint, plus 1 summary grid (5×5 montage). Total ~26 PNGs per checkpoint, ~126 MB per set.

**Storage locations** (DVC-tracked under `outputs/visualizations.dvc`):
- `outputs/visualizations/val_cam_exploration/` — MCTformer MC115 baseline
- `outputs/visualizations/spdnet_val_cam_exploration/` — SPDNet broken model (for reference)
- `outputs/visualizations/spdnet_n1_best/` — SPDNet N=1 epoch 69
- `outputs/visualizations/spdnet_n3_best/` — SPDNet N=3 epoch 53
- `outputs/visualizations/spdnet_n3_last/` — SPDNet N=3 epoch 56

#### 5.7.6 SPDNet Overfitting Analysis

Overfitting was the dominant problem across SPDNet experiments:

| Run | Augmentation | Train mAP (final) | Val mAP (best) | Gap | Diagnosis |
|-----|-------------|-------------------|----------------|-----|-----------|
| n1_heavy | heavy | 0.889 | 0.859 | 3.0pp | Healthy |
| n3_heavy | heavy | 0.965 | 0.898 | 6.7pp | Moderate overfit |
| n3_light | light | 0.990 | 0.894 | 9.6pp | Significant overfit |
| n3_minimal | minimal | 0.999 | 0.821 | 17.8pp | Severe overfit |

**Pattern**: more references (N=3 vs N=1) + weaker augmentation → worse overfitting. The N=3 model has more capacity to memorize training pairs. Without strong augmentation, it exploits reference-query correlations that don't generalize.

**Unexplored mitigation strategies**:
- Label smoothing (not implemented for SPDNet; was removed from config as unused)
- Mixup/CutMix at the pair level
- Reducing backbone capacity (ResNet34, MobileNet)
- Higher dropout (currently uses backbone default, no explicit dropout added)
- Freezing early backbone layers

---

## 5.8 Backbone Feature Seeds & CRF Sweep Discovery (April 2026)

### 5.8.1 Motivation: Classifier Projection Destroys Spatial Information

Quantitative analysis during SPDNet investigation revealed that the classifier linear projection (`self.classifier.weight @ fused_features`) dramatically reduces localization quality. Raw backbone FPN features (256 channels) contain significantly richer spatial information than the projected per-class CAMs.

**Key insight**: The classifier learns to map 256-dimensional features → class logits. This projection optimizes for *classification*, not *localization*. Spatial patterns that don't correlate with class discrimination are suppressed.

### 5.8.2 Feature Seed Types Evaluated

Three seed extraction methods were implemented and compared against the standard ADPL-CAM approach:

| Seed Mode | Description | Aggregation |
|-----------|-------------|-------------|
| `feat_chmean` | Channel-mean of merged FPN query features | `features.mean(dim=1)` → (H,W) |
| `feat_chmax` | Channel-max of merged FPN query features | `features.amax(dim=1)` → (H,W) |
| `cam_max` | Standard ADPL-CAM with max binary aggregation | Existing pipeline |
| `spatial_proto` | Cosine similarity between query features and reference prototype | `F.normalize(query) · F.normalize(ref_proto)` |

All methods use multi-scale (1.0, 0.75, 1.25) + horizontal flip augmentation, averaging the resulting 2D maps across scales. Feature maps are reduced from 256-ch to 1-ch **on GPU** before CPU transfer, avoiding OOM.

### 5.8.3 Quantitative Results (200 val images, n1_best checkpoint)

**Threshold sweep results (direct binary mask, no CRF):**

| Seed Mode | Best Threshold | Disease IoU | BG IoU | mIoU |
|-----------|---------------|-------------|--------|------|
| `feat_chmean` | 0.35 | **36.50%** | 78.27% | **57.39%** |
| `cam_max` (ADPL-CAM) | 0.12 | 17.84% | 69.21% | 43.53% |
| `spatial_proto` | 0.00 | 17.60% | 0.00% | 8.80% |

**Key finding**: Backbone features (`feat_chmean`) achieve **+18.66pp disease IoU** over classifier-projected CAMs (+13.86pp mIoU). This is a dramatic improvement from simply skipping the classification projection.

### 5.8.4 CRF Parameter Sweep Results

CRF was applied to `feat_chmean` seeds with a grid of 60 configurations (4 srgb × 5 bg_threshold × 3 scale_factor values). Top 5 results:

| srgb | bg_threshold | scale_factor | Disease IoU | BG IoU | mIoU |
|------|-------------|-------------|-------------|--------|------|
| 5 | 0.30 | 1.0 | **42.13%** | 79.61% | **60.87%** |
| 8 | 0.30 | 1.0 | 41.95% | 79.86% | 60.91% |
| 3 | 0.30 | 1.0 | 41.72% | 79.05% | 60.39% |
| 13 | 0.30 | 1.0 | 41.49% | 79.92% | 60.71% |
| 13 | 0.30 | 6.0 | 36.88% | 75.29% | 56.09% |

**CRF findings**:
1. **CRF adds +5.63pp disease IoU** over raw thresholded `feat_chmean` (42.13% vs 36.50%)
2. **`bg_threshold=0.30`** dominates all top configs — lower thresholds oversegment disease
3. **`scale_factor=1.0`** (no PSA scaling) works best — higher scaling tightens boundaries too aggressively
4. **`srgb=5` is optimal** (vs VOC default of 13) — lower color bandwidth makes CRF more sensitive to disease color differences, confirming that plant disease spots need tighter color kernels
5. The srgb effect is modest (42.13 vs 41.49 between best and default), suggesting the spatial unary potentials dominate

### 5.8.5 Spatial Prototype Analysis

The spatial prototype matching (`spatial_proto`) tests whether reference-query cosine similarity can guide localization. Results show it does NOT outperform simple channel-mean:
- `spatial_proto` achieves 17.60% disease IoU at threshold=0.00 (everything is predicted as disease)
- The cosine similarity map has a narrow value range, making it hard to threshold
- **Conclusion**: The current SPDNet reference fusion (GlobalMaxPool → channel-only offset) loses all spatial information from references. Spatial prototype matching at inference cannot recover what was never learned.

### 5.8.6 Summary: Pipeline Comparison

| Method | Disease IoU | Improvement |
|--------|-------------|-------------|
| MCTformer CAM (baseline) | 29.98% | — |
| SPDNet ADPL-CAM (n1_best, max-agg) | 28.08% | -1.90pp |
| SPDNet feat_chmean (this work) | 36.50% | +6.52pp |
| SPDNet feat_chmean + CRF(srgb=5) | **42.13%** | **+12.15pp** |

### 5.8.7 Implementation Details

**New files:**
- `src/wsss/spdnet/model.py`: Added `extract_merged_features()`, `_get_fpn_features()`, `_merge_fpn()` methods
- `src/wsss/spdnet/cam_generator.py`: Added `generate_spdnet_seed()` and `generate_all_seeds()` functions
- `src/generate_spdnet_cams.py`: Added `seed_mode` config parameter
- `src/wsss/refinement/crf.py`: Exposed `srgb` parameter (was hardcoded to 13)
- `src/apply_crf.py`: Pass `srgb` through from `CRFConfig`
- `scripts/sweep_crf_params.py`: CRF parameter grid search script
- `scripts/evaluate_feature_seeds.py`: End-to-end evaluation orchestrator
- `tests/test_spdnet.py`: Added 7 new tests (TestFeatureSeeds, TestCRFSrgb, TestExtractMergedFeatures)

**Best CRF config saved to**: `outputs/spdnet_plantseg/feature_seed_eval/best_crf_config.json`

**Reproduction:**
```bash
export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

# Full evaluation (200 images, ~1h)
python scripts/evaluate_feature_seeds.py \
    --checkpoint outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt \
    --max_images 200

# Just CRF sweep on existing seeds
python scripts/sweep_crf_params.py \
    --seed_dir outputs/spdnet_plantseg/feature_seed_eval/seeds_feat_chmean \
    --image_dir data/plantsegv3/images/val \
    --gt_dir outputs/plantseg_binary_mc115/gt_binary_val \
    --max_images 200 --num_workers 8

# Unit tests
python -m pytest tests/test_spdnet.py -v
```

---

## 6. Key Technical Decisions & Rationale

### 6.1 Binary Framing
116-class supervised segmentation achieved only 44.2% mIoU. Binary framing (bg vs disease) simplifies the task dramatically while remaining useful for disease detection applications.

### 6.2 MCTformer-V2 over Traditional Classifiers
GradCAM from ResNet/EfficientNet produced CAM IoU ~20%. MCTformer's class/patch token interaction generates significantly better-localized activation maps, making it the preferred classifier for WSSS.

### 6.3 Max Aggregation for Multiclass->Binary CAM Conversion
Options considered:
- `max`: preserves the strongest activation signal from any disease class (chosen)
- `mean`: dilutes strong signals with weak/absent ones
- `top-K mean`: compromise, but adds a hyperparameter
- `softmax-weighted mean`: theoretically sound but complex

`max` was selected as the most robust default, preserving strong localized signals without dilution.

### 6.4 PlantVillage Augmentation for MC115 Training
PlantVillage provides ~54k image-level-labeled samples (vs ~7.7k in PlantSeg). Adding PV to MCTformer training improves classification performance for rare diseases and encourages the model to learn more discriminative features.

### 6.5 GT-Mask Removal in CAM Generation
The original `generate_cams.py` multiplied attention maps by GT labels (`cls_att * target.view(...)`), zeroing out activations for non-GT classes. This was correct for the standard WSSS pipeline (generate class-specific CAMs) but harmful for the binary aggregation approach (discards potentially useful cross-class disease signals). The fix was to remove this masking when `binary_aggregate` is set.

---

## 7. MLflow Experiments (Full Index)

| ID | Name | Runs | Period | Domain |
|----|------|------|--------|--------|
| `358457224855191874` | plantseg_architecture_benchmark | 7 | Nov-Dec 2025 | Supervised seg |
| `591324269892947917` | plantseg_augmentation_ablation_fp32_final | 13 | Dec 2025 | Supervised seg |
| `319358957156901464` | plantseg_multiclass_benchmark | 1 | Dec 2025 | Supervised seg |
| `675534359741067840` | dfbakin_classifier_cam_benchmark | 3 | Jan 2026 | CAM quality |
| `229846199444711023` | mctformer_voc_v2 | 1 | Feb 2026 | WSSS (VOC) |
| `803960624544164050` | dfbakin_mctformer_voc | 2 | Feb 2026 | WSSS (VOC) |
| `274461605482018881` | weakclip_voc | 1 | Feb 2026 | WSSS (VOC) |
| `891199103211063086` | weakclip_voc_scheduler | 1 | Feb 2026 | WSSS (VOC) |
| `751573413955258358` | weakclip_voc_fixed | 2 | Mar 2026 | WSSS (VOC) |
| `482268093444064376` | weakclip-voc-fixed | 2 | Mar 2026 | WSSS (VOC) |
| `301684071360501862` | weakclip-voc | 4 | Mar 2026 | WSSS (VOC) |
| `826207548213372587` | deeplab_wsss_voc | 1 | Feb 2026 | WSSS (VOC) |
| `440861154593622755` | deeplab_wsss_smoke | 1 | Feb 2026 | WSSS (VOC) |
| `771976166740944756` | mctformer_plantseg | 2 | Mar 2026 | WSSS (PlantSeg) |
| `592872031317447660` | mctformer_plantseg_binary | 2 | Mar 2026 | WSSS (PlantSeg) |
| `247242759685982163` | smoke_mctformer_binary | 5 | Mar 2026 | WSSS (PlantSeg) |
| `206029539984518366` | weakclip-plantseg-binary | 2 | Mar 2026 | WSSS (PlantSeg) |
| `444840611733456632` | weakclip-plantseg-binary-t_0.64 | 1 | Mar 2026 | WSSS (PlantSeg) |
| `113502554219633766` | weakclip-plantseg-binary-t_0.73 | 1 | Mar 2026 | WSSS (PlantSeg) |
| `285465004951754042` | spdnet_plantseg | 5+ | Apr 2026 | SPDNet Siamese (PlantSeg) |

Runs within `spdnet_plantseg` experiment:
- `spdnet_448_42` — broken model (ref ignored), 45 ep, mAP=0.621
- `spdnet_fix_n1_heavy` — Token N=1, heavy aug, 80 ep, mAP=0.859 (used as token baseline in 5.10)
- `spdnet_fix_n3_heavy` — Token N=3, heavy aug, 57 ep (crash), mAP=0.898
- `spdnet_fix_n3_light` — Token N=3, light aug, 80 ep, mAP=0.894
- `spdnet_fix_n3_minimal` — Token N=3, minimal aug, 80 ep, mAP=0.821
- `spdnet_spatial_n1_ps` — Spatial cross-attn, N=1, PlantSeg only, 80 ep, mAP=0.797, gate=0.333
- `spdnet_spatial_n1_ps_pv` — Spatial cross-attn, N=1, PlantSeg+PV, 80 ep, mAP=0.888, gate=0.499
- 1-epoch spatial smoke run (PlantSeg+PV pipeline check) — kept in MLflow only
- Smoke/test runs: `test_e2e`, `test_n3` (deleted from disk, may remain in MLflow)

---

## 8. Infrastructure & Tooling

| Tool | Version / Details | Role |
|------|------------------|------|
| **Python** | 3.11 | Runtime |
| **PyTorch** | 2.9.1 | Deep learning framework |
| **Lightning** | 2.5.6 | Training framework |
| **Hydra** | 1.3.2 | Configuration management (all scripts use `@dataclass` configs) |
| **MLflow** | 3.6.0 | Experiment tracking (local file store) |
| **DVC** | (in venv) | Data & artifact versioning (S3 remote) |
| **timm** | 1.0.22 | Pretrained model zoo |
| **transformers** | 4.57.1 | SAM model loading |
| **albumentations** | 2.0.8 | Image augmentation |
| **Ruff** | - | Linter (PEP8 + import sorting) |
| **Quarto** | - | Report generation (beamer presentations) |
| **Poetry** | - | Dependency management |

### DVC-tracked artifacts
```
data/VOC2012.dvc              # Pascal VOC dataset
data/plant-village.dvc        # PlantVillage dataset
data/plantsegv3.dvc           # PlantSeg V3 dataset
data/plant-pathology-2020-fgvc7.dvc  # Plant Pathology (auxiliary)
mlruns.dvc                    # All MLflow experiments (incl. spdnet_plantseg
                              #   + spatial token / spatial PS / spatial PS+PV runs)
pretrained.dvc                # Pretrained weights (CLIP, ResNet-38)
reports/resources.dvc          # Report figures/resources
outputs/plantseg_binary/*.dvc  # Binary pipeline artifacts
outputs/plantseg_binary_mc115/*.dvc  # MC115 pipeline artifacts
outputs/spdnet_plantseg.dvc   # SPDNet checkpoints, ADPL-CAMs AND
                              #   spatial training checkpoints + corrected-refs
                              #   feat_chmean/feat_chvar/cam_classifier seeds
                              #   (~34 GB; the full-val seed dirs are the most
                              #    expensive artifact in the repo — ~10h compute each)
outputs/visualizations.dvc    # All activation visualizations including spatial
                              #   attention grids and corrected-refs CRF viz (~1.5 GB)
```

**Most computationally expensive artifacts** (ranked, all under `outputs/spdnet_plantseg.dvc`):
1. `spdnet_*_eval/seeds_cam_classifier_max_corrected_refs/` (3 × ~3 GB) — **the closest thing to "CAMs"** for the spatial models; require trained checkpoint + per-image forward + multi-scale + flip + classifier projection on fused features.
2. `spdnet_*_eval/seeds_feat_chvar_corrected_refs/` (3 × ~3 GB) — channel variance maps from pre-fusion features.
3. `spdnet_*_eval/seeds_feat_chmean_corrected_refs/` (3 × ~3 GB) — channel mean maps.
4. `spdnet_spatial_n1_ps_pv/checkpoints/` (~620 MB) — 13h spatial training on PS+PV.
5. `spdnet_spatial_n1_ps/checkpoints/` (~620 MB) — 2.5h spatial training on PS.

Each of items 1–3 took roughly 4–8 hours of GPU time (1247 images × 3 models). Re-running them costs an overnight per category.

**DVC remote**: `s3://plant-diseases-bucket/dvc_dir` (Yandex Object Storage, `ru-central1`). Config in `.dvc/config.local` (credentials) and `.dvc/config` (public read URL). Use `export PATH="/venv/main/bin:$PATH"` before DVC commands.

**To restore SPDNet + spatial artifacts on a new machine**:
```bash
dvc pull outputs/spdnet_plantseg.dvc outputs/visualizations.dvc mlruns.dvc
```

### Git Branches
| Branch | Purpose |
|--------|---------|
| `master` | Stable base |
| `wsss-weakclip-pipeline` | **Current**: WSSS pipeline development (MC115, SAM, etc.) |
| `arch-benchmark-exps` | Architecture benchmark experiments |
| `augmentation-ablation` | Augmentation ablation experiments |
| `disease-classifier` | Classifier-CAM benchmark |
| `multiclass-segmentation` | Multiclass supervised segmentation |
| `segnext-model-add` | SegNeXt model integration |
| `scheduler-test` | Learning rate scheduler experiments |

---

## 9. Reports (Quarto Beamer Presentations)

| File | Title | Content |
|------|-------|---------|
| `01_datasets_ideas.qmd` | Datasets & Ideas | Overview of PlantSeg, PlantVillage, research directions |
| `02_architecture_benchmark.qmd` | Architecture Benchmark | SegFormer vs SegNeXt vs UNet vs DeepLabV3+ |
| `03_classifier_cam_benchmark.qmd` | Classifier-CAM Benchmark | GradCAM quality: ResNet vs EfficientNet |
| `05_mask_quality_analysis.qmd` | Pseudomask Quality Analysis | Binary vs MC115 pipeline comparison with diagnostics |
| `06_spdnet_spatial_findings.qmd` | SPDNet Spatial Findings | SPDNet architecture with seed extraction Points A/B/C, spatial cross-attention, corrected-refs evaluation, and final verdict (rendered to `06_spdnet_spatial_findings.pdf`) |

---

## 10. Open Problems & Future Directions

1. **SAM refinement — geometric prompts are viable**: Experiment J (bbox + points) is the only SAM configuration that works. Dense mask prompts (binary or soft) fail catastrophically. Future work should explore: (a) optimal point count/quantile trade-off (J's q=0.90 with 5 points beat the tuned q=0.99 with 20 points), (b) iterative SAM refinement (use J output as new bbox source for a second pass), (c) combining J output with CRF post-processing.

2. **SAM1 / GroundedSAM text prompts**: SAM1 checkpoints do not support text prompts natively. GroundedSAM (combining Grounding DINO + SAM) could enable text-prompted disease segmentation. Deferred to future work.

3. **CAM localization quality is the fundamental ceiling — BUT bypassing the classifier helps**: Point sampling analysis shows only ~53% of positive points (at q=0.99) land on actual disease. **SPDNet ADPL-CAM (28.08% disease IoU) did not surpass MCTformer (29.98%)**, confirming that *classifier-projected* CAMs are a bottleneck. However, **raw backbone features (feat_chmean)** bypass the classifier entirely and achieve **36.50% disease IoU** (+6.52pp over MCTformer), reaching **42.13%** with CRF refinement (srgb=5, bg_threshold=0.30). This proves the localization information exists in the network but is destroyed by the classification projection.

4. **Disease-IoU-optimized thresholds**: The threshold sweep fix (optimizing for disease IoU instead of mIoU) recovers +2.58pp. This should be propagated to all pipeline stages (CRF, PSA+RW binarization).

5. **MCTformer backbone is fully fine-tuned**: The ViT backbone (DeiT-Small) is not frozen during MCTformer training. Freezing early layers or using LoRA-style adaptation could reduce overfitting and potentially improve CAM localization, especially for the 115-class task.

6. **Cross-dataset transfer**: PlantVillage → PlantSeg transfer for WSSS is the ultimate goal. The current pipeline trains on PlantSeg annotations but the WSSS approach would enable training on PlantVillage (image-level only) and deploying on PlantSeg-like images.

7. **Aggregation strategies**: `max` outperformed `top_energy` in both MCTformer and SPDNet experiments. `top-K mean` or `softmax-weighted mean` remain untested but are unlikely to change the picture significantly.

8. **Stronger classifier backbones**: MCTformer-V2 uses DeiT-Small. Upgrading to DeiT-Base or using more recent ViT variants could improve CAM quality.

9. **End-to-end evaluation**: Training a final segmentation model (DeepLabV3+, SegFormer) on the WSSS pseudo-masks and evaluating on a held-out test set would give the most meaningful quality assessment.

10. **SPDNet remaining experiments**: N=5 and N=8 reference runs were not executed. Given that N=3 showed *worse* CAM localization than N=1 (despite better classification), higher N is unlikely to improve CAMs. Scripts are ready in `scripts/run_spdnet_experiments.sh` (set `RUNS="3 4"` for N=5/N=8).

11. **SPDNet overfitting mitigation**: All N=3 runs showed 6.7–17.8pp train-val gap. Unexplored strategies: label smoothing, mixup/cutmix at pair level, reduced backbone capacity (ResNet34/MobileNet), explicit dropout after FPN, freezing early backbone layers.

12. **SPDNet inference strategy**: For deployment without GT class labels, a two-stage approach: (1) MCTformer single-shot classification (77% accuracy, fast) predicts disease class, (2) sample K reference images from predicted class, (3) SPDNet forward pass generates ADPL-CAM. Not yet implemented.

13. **Hybrid CAM ensemble**: Element-wise max or weighted average of MCTformer attention-based CAMs + SPDNet ADPL-CAMs could capture complementary signals. Not yet tested — a low-cost experiment since both CAM sets already exist on disk.

14. **Feature seed full-dataset evaluation**: The 42.13% disease IoU from feat_chmean+CRF was measured on 200 val images. Running on all 1247 val images would give more reliable metrics and could be used as PSA training input.

15. **Feature seed → PSA → Random Walk pipeline**: The feat_chmean seeds are continuous float maps compatible with the full refinement pipeline (CRF → PSA → Random Walk → DeepLab). Testing this end-to-end could yield significantly better pseudo-masks than the current ADPL-CAM-based pipeline.

16. **Spatial reference with trained attention (IMPLEMENTED & EVALUATED — April 2026)**: A `SpatialCrossAttention` module was implemented (`fusion_mode="spatial"`) that preserves spatial structure via multi-head cross-attention. Trained on PlantSeg-only (val mAP=79.7%, gate=0.333) and PlantSeg+PV (val mAP=88.8%, gate=0.499). **Outcome**: spatial fusion did NOT improve localization — `cam_classifier` DisIoU was 30.4–31.0% (CRF) vs token baseline's 32.5% (CRF). Cross-attention visualizations show a query-invariant pattern with no alignment to reference disease regions. The architecture has the capacity but classification-only training does not provide a learning signal for spatial discrimination. Detailed analysis in Sections 5.9–5.10.

17. **Feature seed from MCTformer backbone**: If backbone features outperform classifier CAMs for SPDNet, the same may hold for MCTformer. Extracting ViT intermediate features (attention-weighted patch tokens) before the CLS→CAM projection could yield better MCTformer seeds.

18. **SPDNet on PlantSeg + PlantVillage (DONE — April 2026)**: All previous token-fusion SPDNet training used PlantSeg only. The two spatial fusion runs include both a PS-only and a PS+PV variant. The PS+PV spatial run reaches val mAP 0.888 (best of any SPDNet config), but its localization metrics are not better than the token PS-only baseline (Section 5.10).

19. **Add explicit spatial supervision to SPDNet (THOUGHT — April 2026)**: Given the negative result above, the natural next step is to add an auxiliary loss (contrastive, equivariance, or self-distillation) so that the cross-attention is forced to be spatially discriminative rather than acting as a content-free bias. Detailed sketch in Section 5.11. Not yet planned for implementation.

---

### 5.9 Spatial Cross-Attention Fusion (April 2026)

**Motivation**: The `spatial_proto` experiment (Section 5.8) revealed that SPDNet's ADPL-CAM reference fusion via `GlobalMaxPool → broadcast add` is *spatially inert* — every query location receives the same additive reference signal. The hypothesis is that a fusion mechanism preserving spatial structure from the reference could enable the model to learn "this location in the query looks similar to a diseased region in the reference."

**Architecture**: `SpatialCrossAttention` replaces `ADPLCam.tokenize + .fuse` with:
1. Merge reference FPN levels to finest resolution (same as query merge)
2. Pool merged reference to 14×14 via `AdaptiveAvgPool2d` → 196 KV tokens
3. Flatten query to HW Q tokens, apply `LayerNorm` on both Q and KV
4. Multi-head cross-attention (4 heads, 256-d) from query→reference
5. Gated residual: `output = query + gate * attended` (gate init 0.1)

Memory cost: O(HW × 196) ≈ 2.5M entries for 112×112 query — negligible. Parameter overhead: ~260K (~1% of 27M total).

**Configuration**: `model.fusion_mode` in `SPDNetModelConfig` selects `"token"` (original ADPL-CAM) or `"spatial"` (cross-attention). Backward-compatible: old checkpoints default to `"token"`.

**Implementation files**:
- `src/wsss/spdnet/model.py`: `SpatialCrossAttention` class, `SPDNet` gains `fusion_mode` parameter
- `src/conf/spdnet.py`: `fusion_mode` field in `SPDNetModelConfig`
- `src/wsss/spdnet/lightning.py`: passes `fusion_mode` to `SPDNet`
- `src/train_spdnet.py`: reads config, logs `fusion_mode` as MLflow tag
- `src/wsss/spdnet/cam_generator.py`: `load_spdnet_from_checkpoint` auto-detects `fusion_mode` from saved hparams

**Experiments** (`scripts/run_spdnet_spatial_experiments.sh`):
- Run 0: 1-epoch smoke test on PlantSeg+PV (validates pipeline before overnight training)
- Run 1: `spdnet_spatial_n1_ps` — spatial fusion on PlantSeg only, 80 epochs, N=1, heavy aug
- Run 2: `spdnet_spatial_n1_ps_pv` — spatial fusion on PlantSeg+PlantVillage, 80 epochs, N=1, heavy aug

#### 5.9.1 Spatial Cross-Attention Training Results

Both spatial runs completed successfully (smoke test passed; full runs trained for 80 epochs each).

| Run | Dataset | N refs | Aug | Best epoch | val/mAP | Learned `gate` | Notes |
|-----|---------|--------|-----|-----------|---------|----------------|-------|
| `spdnet_token_n1_heavy` (baseline) | PlantSeg only | 1 | heavy | 69 | **0.859** | n/a (token) | ADPL-CAM token fusion |
| `spdnet_spatial_n1_ps` | PlantSeg only | 1 | heavy | 76 | **0.797** | **0.333** | Spatial cross-attention |
| `spdnet_spatial_n1_ps_pv` | PlantSeg + PlantVillage | 1 | heavy | 76 | **0.888** | **0.499** | Spatial cross-attention, larger data |

**Key training observations**:
1. **Spatial fusion converges**: The `gate` parameter (init 0.1) grew during training to **0.333** (PS-only) and **0.499** (PS+PV), confirming the spatial residual contributes non-trivially to the classifier output. It is not pruned to zero — the model genuinely uses the cross-attention pathway.
2. **PS-only spatial underperforms token baseline by 6.2pp mAP** (0.797 vs 0.859). The added cross-attention parameters and modified gradient path apparently make optimization slightly harder on the smaller dataset.
3. **PS+PV spatial outperforms token baseline by 2.9pp mAP** (0.888 vs 0.859). The combined dataset (~41K images vs 7.7K) absorbs the extra capacity well — the larger gate (0.499) suggests the model relies more heavily on reference signal here.
4. Wall-clock times were close to estimates (~2.5h for PS-only, ~13h for PS+PV on RTX 5090).

**Initial hypothesis (test result)**: Better localization via spatial fusion. **Status**: not confirmed — see Section 5.10 for the full evaluation and verdict.

---

### 5.10 Reference Selection Bug Fix & Final Spatial Evaluation (April 2026)

#### 5.10.1 The Reference Selection Bug

Visual inspection of `visualize_spatial_attention.py` outputs revealed that for an **apple** query image the script was pairing it with a **maple-leaf** reference. Investigation traced the issue to the construction of `label_dict` in the evaluation/visualization scripts:

- During training, `SiamesePlantSegDataset` correctly samples **same-class** references using PlantSeg per-image labels.
- During evaluation, `eval_spatial_runs.py` (and earlier evaluation drivers) loaded each validation image with a placeholder `class 0` label because no per-image label file was being read. As a result, `cam_generator.generate_all_cams` was sampling references uniformly from the PlantSeg train set, **regardless of the query class**.

**Impact**:
- All previously reported "spatial" localization metrics (those without `_corrected_refs` in the path) were measured under random-class reference selection — a distribution shift relative to training.
- The token model (which is by construction reference-blind for `feat_chmean`/`feat_chvar` because they read **pre-fusion** features) was unaffected for those modes.
- For modes that *do* depend on the reference at inference (`cam_classifier`, attention visualization), the bug masked the spatial fusion's true behavior.

#### 5.10.2 The Fix

A new helper module was added: `src/wsss/spdnet/class_resolver.py`.

- `load_class_names()` — read the canonical 115-class names list.
- `make_filename_class_resolver()` — build an `image_stem → class_index` mapping using **longest-prefix matching** against class names (e.g., `apple_black_rot_28.jpg` → `apple_black_rot`).
- `build_class_pool_from_labels()` — scan train labels and produce a `Dict[class_idx, List[image_name]]` filtered by physical existence on disk.

`src/wsss/spdnet/cam_generator.py:generate_all_seeds()` and `generate_all_cams()` now accept three new optional arguments:
- `ref_pool: Dict[int, List[str]]`
- `ref_image_dir: Path` (typically `data/plantsegv3/images/train`)
- `query_class_resolver: Callable[[str], int | None]`

When provided, references are sampled from `ref_pool[query_class]` inside `ref_image_dir`, exactly mirroring the training distribution.

`scripts/eval_spatial_full.py`, `scripts/eval_cam_classifier.py`, and `scripts/visualize_spatial_attention.py` were updated to build the resolver/pool at startup and to write all results into `_corrected_refs`-suffixed directories so that buggy and corrected runs coexist for comparison.

**Verification**:
- `scripts/smoke_test_ref_fix.py` — patches `PIL.Image.open` to log every image load and asserts query→ref class equality across 8 queries from different classes. Passed.
- `scripts/smoke_test_eval_pipeline.py` — runs `feat_chmean`/`feat_chvar` end-to-end on 5 images per model. Passed.
- `scripts/smoke_test_cam_classifier.py` — additionally checks that `cam_classifier` outputs differ across token vs spatial models (Pearson correlation 0.61–0.91, vs ~1.0 for `feat_chmean`/`feat_chvar` which read pre-fusion features). Passed.

#### 5.10.3 Per-Distribution CRF Tuning Methodology

After the bug fix, every (model, seed_mode) combination has its own seed value distribution. Reusing CRF parameters tuned on a different distribution biased the comparison. The new evaluation pipeline therefore performs **per-distribution CRF tuning**:

1. Generate seeds for the full val set (1247 images).
2. Sweep binarization thresholds (0.00–0.99 step 0.01) and pick the threshold that maximizes disease IoU.
3. Sweep CRF parameters (4 srgb × 5 bg_threshold × 3 scale_factor = 60 configs) on a 200-image subset, picking the configuration that maximizes disease IoU.
4. Run the chosen CRF on the full val set and report final metrics.
5. Generate visualizations using the per-distribution-best CRF mask in teal/cyan with magenta GT contours.

Results are written to `outputs/spdnet_plantseg/<run>_eval/` and aggregated in:
- `eval_summary_corrected_refs.json` — feat-based seeds
- `eval_summary_cam_classifier.json` — classifier-projected CAMs

#### 5.10.4 Feature-Based Seed Results (Corrected References, Full Val Set)

`feat_chmean` and `feat_chvar` are computed from **pre-fusion** query features (`query_merged`, "Point A" in the architecture diagram). They are therefore **insensitive to the fusion mode** by construction — any small differences come solely from gradient-induced changes to the backbone weights during training.

| Run | Seed mode | Best thr | DisIoU (thr) | CRF params | DisIoU (CRF) | BG IoU (CRF) | mIoU (CRF) |
|-----|-----------|---------:|-------------:|-----------|-------------:|-------------:|-----------:|
| Token N=1 PS (baseline) | `feat_chmean`* | 0.33 | 36.35% | s=8 bg=0.30 sc=1 | **38.32%** | 70.66% | 54.49% |
| Token N=1 PS (baseline) | `feat_chvar`   | 0.11 | 35.24% | s=13 bg=0.10 sc=1 | 37.63% | 71.22% | 54.42% |
| Spatial PS-only | `feat_chmean` | 0.00 | 17.87% | s=13 bg=0.05 sc=1 | 20.33% | 0.00% | 10.17% |
| Spatial PS-only | `feat_chvar`* | 0.13 | 34.21% | s=13 bg=0.10 sc=1 | **36.94%** | 67.39% | 52.17% |
| Spatial PS+PV  | `feat_chmean` | 0.00 | 17.87% | s=8 bg=0.05 sc=1 | 20.33% | 0.00% | 10.16% |
| Spatial PS+PV  | `feat_chvar`* | 0.06 | 34.08% | s=13 bg=0.05 sc=1 | **37.15%** | 67.86% | 52.51% |

(*) marks the best seed mode per run.

**Observations**:
- For the spatial models, `feat_chmean` collapses (DisIoU ≈ 17.9%, BG IoU = 0%): the channel-mean of the trained spatial backbone is **negative-skewed**, so almost everything is below threshold 0 and gets predicted as disease. `feat_chvar` (channel variance) sidesteps this by being scale-invariant and yields competitive numbers.
- Best spatial result: **37.15% DisIoU (CRF)** with `feat_chvar` on PS+PV — **−1.17pp below the token baseline** (`feat_chmean` 38.32%).
- Spatial PS-only `feat_chvar` (36.94%) is essentially tied with spatial PS+PV (37.15%), even though the latter has +9.1pp higher classification mAP. Localization quality of pre-fusion features is dominated by the backbone, not by the dataset size.

#### 5.10.5 cam_classifier Results (the Real Test of Spatial Fusion)

`cam_classifier` projects the trained classifier weights onto the **fused features** ("Point C" in the architecture diagram). For the spatial model this is the only seed mode that actually exercises the cross-attention pathway. It is the correct apples-to-apples comparison.

| Run | Fusion | Gate | Best thr | DisIoU (thr) | CRF params | DisIoU (CRF) | BG IoU (CRF) | mIoU (CRF) |
|-----|--------|------|---------:|-------------:|-----------|-------------:|-------------:|-----------:|
| Token N=1 PS | token | n/a | 0.25 | 29.10% | s=3 bg=0.20 sc=1 | **32.49%** | 68.60% | 50.55% |
| Spatial PS-only | spatial | 0.333 | 0.14 | 31.00% | s=13 bg=0.15 sc=1 | 30.37% | 74.02% | 52.19% |
| Spatial PS+PV | spatial | 0.499 | 0.21 | 28.49% | s=8 bg=0.20 sc=1 | 30.98% | 69.26% | 50.12% |

**Reference (historical baselines)**:
- MCTformer MC115 (raw CAM, disease-IoU optimized, 500 imgs): ~29.98% DisIoU.
- MCTformer MC115 (HA-CRF, full val): 31.16% DisIoU.

**Verdict**: After per-distribution CRF tuning on the full val set:
- The **token baseline reaches 32.49% DisIoU** — the strongest cam_classifier result.
- Both **spatial variants land 1.5–2.1 pp BELOW** the token baseline on disease IoU even though they have a non-trivial learned gate.
- mIoU is essentially flat (50.12–52.19%) because losses on disease are recouped on background.

Combined with the cross-attention visualizations (Section 5.10.6), this is strong evidence that **the spatial cross-attention learnt by classification-only training does not produce a useful localization signal**.

#### 5.10.6 Cross-Attention Visualization Findings

`scripts/visualize_spatial_attention.py` produces multi-panel grids per query: head-averaged attention map, per-query-location attention onto the reference grid, and the softmax distribution across the 196 reference tokens.

Findings (with corrected same-class references on PlantSeg val):
- The head-averaged attention map is **largely query-invariant**: most queries attend to the same handful of reference tokens (typically corner/border positions of the 14×14 grid) rather than reference disease regions.
- There is no visible alignment between high-attention reference locations and the disease spots highlighted by GT in the reference image.
- The pattern is consistent across both spatial models (`gate=0.333` and `gate=0.499`) — the learned attention is acting more like a *position-conditioned bias* than a content-driven correspondence.

This explains why the gate is non-trivial (the residual is useful for classification — it's a free bias term) yet localization does not improve: the cross-attention has learnt to inject a **constant-ish residual** into query features rather than spatially-targeted reference evidence.

#### 5.10.7 Final Verdict on Spatial Cross-Attention as Implemented

| Question | Answer |
|----------|--------|
| Did the spatial residual stay alive during training? | Yes — gate grew from 0.1 → 0.33 / 0.50 |
| Did spatial fusion improve classification mAP? | Mixed — −6.2pp on PS-only, +2.9pp on PS+PV |
| Did spatial fusion improve `feat_chmean`/`feat_chvar` localization? | No — `feat_chvar` ≈ token baseline; `feat_chmean` collapses |
| Did spatial fusion improve `cam_classifier` localization? | **No** — 30.4–31.0% DisIoU vs token's 32.5% |
| Does the attention align with disease regions in the reference? | No — query-invariant pattern, no GT alignment |

**Conclusion**: Adding the spatial cross-attention module is **necessary but not sufficient**. The architecture has the capacity to do spatial reference matching, but classification-only training does not provide a learning signal that forces the attention to be spatially discriminative. The Siamese hypothesis is **not yet rejected** — it is rejected *given the current training objective*. To genuinely test it we need to introduce explicit spatial supervision (Section 5.11).

---

### 5.11 Future Direction: Adding Explicit Spatial Supervision (Thought, April 2026)

> **Status**: idea sketch, not yet thought through deeply. Recorded here so it can be revisited.

The empirical finding from Section 5.10 is that the spatial cross-attention pathway is exercised (non-trivial gate) but does not learn a spatially-discriminative attention pattern under classification-only loss. The classifier can extract the bit it needs (a global content boost) without ever forcing the attention to match disease regions.

The natural fix is to **add an auxiliary loss whose gradient flows through the attention map and rewards spatial agreement** with structure that we know correlates with disease localization. Three complementary directions to explore:

1. **Patch-level cross-image contrastive loss**
   - Treat the query and reference as two views of the "same disease" and pull together patch-level features that match disease semantics, while pushing apart non-matching patches.
   - Can use disease class labels (positive pair = same-class query/ref disease patches; negative = different-class or background patches), or a SimCLR-style InfoNCE over the cross-attention logits.
   - Effect: forces the cross-attention to put high weight on reference patches that look like disease in the query.

2. **Equivariance loss (geometric self-consistency)**
   - For augmentation `T` (flip, crop, rotation), enforce `attention(T(query), ref) ≈ T(attention(query, ref))`.
   - Pure-classification training does not need equivariant attention; adding this loss makes the spatial map follow the query's geometric content.
   - Cheap to implement (re-run forward on a transformed query within the same step).

3. **Self-distillation from CAM pseudo-masks**
   - Use the model's own thresholded (or CRF-refined) CAM as a noisy disease pseudo-mask.
   - Distill the cross-attention output to peak inside the pseudo-mask region (e.g., max-pool or KL-style loss between attended-feature heatmap and pseudo-mask).
   - Risk: a fixed-point loop where bad CAMs reinforce themselves. Mitigated by applying it only after a warm-up of N classification epochs and using EMA pseudo-masks.

**Tentative composite loss** (notation only — coefficients and exact formulations TBD):

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + \lambda_1 \, \mathcal{L}_{\text{contrastive}} + \lambda_2 \, \mathcal{L}_{\text{equivariance}} + \lambda_3 \, \mathcal{L}_{\text{self-distill}}
\]

with `λ_3 = 0` for the first warm-up phase to avoid the self-reinforcing degeneracy.

**Open questions to think through before implementing**:
- Choice of "same disease" anchor for the contrastive loss — class labels are coarse, GT masks would be ideal but defeat the WSSS premise. A mid-ground is to use the model's own CAM peaks as initial anchors and refine with EMA.
- Where to apply each loss in the network (on the attention logits, on attended features, or both).
- How to weight the auxiliary losses against classification — too strong and classification suffers; too weak and the attention reverts to the lazy bias pattern observed in Section 5.10.6.
- Whether an equivariance loss alone is sufficient (it is the cheapest and least invasive of the three).

This is the most natural next experiment after the current "spatial fusion does not learn spatial signal" finding. It is **not yet planned for implementation**.

---

## 11. How to Reproduce Key Results

### Binary pipeline
```bash
export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

# Pull data and pretrained weights
dvc pull data/plantsegv3.dvc data/plant-village.dvc pretrained.dvc

# Run full binary pipeline
./scripts/run_plantseg_binary_pipeline.sh

# Or skip early steps if artifacts exist
SKIP_STEPS="0,1,2,3,4,5,6,7" ./scripts/run_plantseg_binary_pipeline.sh
```

### MC115 pipeline
```bash
MCTFORMER_DATASET=plantseg_with_pv \
MCTFORMER_EXPERIMENT=mctformer_plantseg_mc115_pv \
BINARY_AGGREGATE=max \
OUT_BASE=outputs/plantseg_binary_mc115 \
BINARY_BASE=outputs/plantseg_binary \
WEAKCLIP_EXPERIMENT=weakclip-plantseg-binary-mc115-t_0.73 \
scripts/run_plantseg_binary_pipeline.sh
```

### SAM1 refinement experiments
```bash
./scripts/run_sam_refinement_experiments.sh
```

### SPDNet Siamese training
```bash
# Pull SPDNet artifacts (if on new machine)
dvc pull outputs/spdnet_plantseg.dvc outputs/visualizations.dvc

# Single run (N=1, heavy augmentation, 80 epochs, ~2.5h on RTX 5090)
python src/train_spdnet.py run_name=spdnet_fix_n1_heavy \
    data.num_references=1 data.augmentation=heavy \
    trainer.max_epochs=80

# Full experiment sweep (6 runs)
RUNS="1 2 3 4 5 6" ./scripts/run_spdnet_experiments.sh

# Only specific runs (N=5 heavy, N=8 heavy)
RUNS="3 4" ./scripts/run_spdnet_experiments.sh

# Spatial cross-attention fusion experiments (all 3 runs, ~15h)
./scripts/run_spdnet_spatial_experiments.sh

# Skip smoke test, run only the full training
RUNS="1 2" ./scripts/run_spdnet_spatial_experiments.sh
```

### SPDNet spatial evaluation (corrected references, full val set)
```bash
# Full feat_chmean + feat_chvar evaluation (token + spatial × seed_mode)
# - per-distribution CRF tuning, full 1247 val images
# - ~7-8h on RTX 5090, automatically skips already-generated artifacts
./scripts/eval_spatial_overnight.sh

# cam_classifier evaluation (the only seed mode that exercises spatial fusion)
# - same per-distribution CRF tuning, full val
# - ~4.3h, similar skip-if-exists behavior
./scripts/eval_cam_classifier_overnight.sh

# Visualize cross-attention maps for spatial models
python scripts/visualize_spatial_attention.py \
    --checkpoint outputs/spdnet_plantseg/spdnet_spatial_n1_ps_pv/checkpoints/epoch=epoch=76-val_mAP=val/mAP=0.8882.ckpt \
    --num-images 25 \
    --output-dir outputs/visualizations/spatial_attention_n1_ps_pv_FIXED

# Reference-selection smoke tests (run before any new evaluation)
python scripts/smoke_test_ref_fix.py
python scripts/smoke_test_eval_pipeline.py
python scripts/smoke_test_cam_classifier.py
```

### SPDNet CAM generation & evaluation
```bash
# Generate CAMs from a checkpoint (max aggregation, N=1)
python src/generate_spdnet_cams.py \
    checkpoint=outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt \
    output_dir=outputs/spdnet_plantseg/cams/n1_best_max \
    binary_aggregate=max num_references=1

# Generate CAMs (top-energy aggregation)
python src/generate_spdnet_cams.py \
    checkpoint=outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt \
    output_dir=outputs/spdnet_plantseg/cams/n1_best_top_energy \
    binary_aggregate=top_energy num_references=1

# Visualize activations (25 images, 8-panel grids)
python scripts/visualize_spdnet_activations.py \
    --checkpoint outputs/spdnet_plantseg/spdnet_fix_n1_heavy/checkpoints/best.ckpt \
    --cam-dir outputs/spdnet_plantseg/cams/n1_best_max \
    --output-dir outputs/visualizations/spdnet_n1_best \
    --num-images 25

# Run unit tests
pytest tests/test_spdnet.py -v
```

### Generate reports
```bash
cd reports
quarto render src/05_mask_quality_analysis.qmd
```

---

## 12. Known Issues & Gotchas

1. **DVC in venv**: DVC is installed in the venv at `/venv/main/bin/dvc`, not globally. Always use `export PATH="/venv/main/bin:$PATH"` or invoke it directly.

2. **class_names.txt conflict**: When running MC115 pipeline, `export_labels.py` writes a 115-class `class_names.txt` alongside labels. Downstream binary evaluation/WeakCLIP expects a single-class file. Solution: MC115 mode creates a separate `binary_class_names.txt` containing just "disease".

3. **GT-masked CAMs**: The original `generate_cams.py` masks CAMs with ground-truth labels. For binary aggregation, this masking must be disabled (`binary_aggregate` parameter handles this automatically).

4. **PSA backbone**: The PSA network requires `pretrained/res38_cls.pth` (ResNet-38d pretrained on ImageNet). This is DVC-tracked under `pretrained.dvc`.

5. **CLIP pretrained**: WeakCLIP requires `pretrained/ViT-B-16.pt`. Also DVC-tracked.

6. **Memory**: SAM1 (ViT-Huge) requires ~8 GB VRAM. WeakCLIP training with multi-scale inference requires ~24 GB. MCTformer training is lightweight (~4 GB).

7. **Hydra override syntax**: Complex values (lists, paths with special chars) need careful quoting:
   ```bash
   "scales=[1.0,0.75,1.25]"        # list
   "checkpoint='path/to/file'"      # path with slashes
   ```

8. **SPDNet checkpoint paths with `=` signs**: Lightning's `ModelCheckpoint` creates filenames like `epoch=69-val_mAP=0.8594.ckpt`. Hydra misinterprets the `=` in CLI arguments. **Workaround**: create symlinks with simple names (e.g., `ln -s "epoch=69-val_mAP=0.8594.ckpt" best.ckpt`) and pass the symlink path to scripts. The committed checkpoints already have `best.ckpt` / `last.ckpt` symlinks.

9. **SPDNet dataset startup time**: `SiamesePlantSegDataset._build_index()` must scan all training samples to build the class→index mapping. After the fix (reading mask files directly instead of loading images), this takes ~10-15 seconds. The original broken version took several minutes.

10. **SPDNet memory with N>3 references**: Each additional reference adds a full ResNet50+FPN forward pass. N=3 with batch_size=12 uses ~18 GB VRAM on RTX 5090. N=5 requires batch_size=10, N=8 requires batch_size=8 (both with gradient_accumulation=4 to maintain effective batch size). Adjust `data.batch_size` in the Hydra config accordingly.

11. **SPDNet n3_heavy crash at epoch 57**: The `spdnet_fix_n3_heavy` run crashed during checkpoint saving (OOM spike). The best checkpoint (epoch 53) and last checkpoint (epoch 56) were both saved before the crash. If re-running, consider using `trainer.max_epochs=60` or increasing checkpoint save frequency.

12. **SPDNet CAM generation is slow**: Generating ADPL-CAMs for 1247 val images takes ~25-35 minutes per configuration (multi-scale + flip). The `generate_spdnet_cams.py` script has an `eval_sweep_samples` parameter to limit the threshold sweep evaluation (but CAM generation still processes all images). For smoke tests, create a tiny `_smoke_labels.npy` file containing only 5 image names.

13. **SPDNet reference selection bug (FIXED, April 2026)**: For a long stretch the standalone evaluation/visualization scripts assigned every validation image to `class 0` because they did not load per-image labels. As a consequence `cam_generator.generate_all_seeds` / `generate_all_cams` sampled references uniformly across all 115 train classes, producing query/reference mismatches (e.g., apple query paired with maple-leaf reference). The fix is in `src/wsss/spdnet/class_resolver.py` (filename → class index resolver + same-class train ref pool) plus new `ref_pool` / `query_class_resolver` arguments on the cam generator. **All evaluation results in Section 5.10 use the corrected refs.** Earlier results (without the `_corrected_refs` suffix on output paths) used random-class references and should not be used for fair comparison — they are kept on disk only for traceability.

14. **Spatial cross-attention checkpoints have nested checkpoint paths**: Lightning's `ModelCheckpoint` interaction with the `=` in the format string produced directories like `epoch=epoch=76-val_mAP=val/mAP=0.7970.ckpt`. The full evaluation/visualization scripts already encode the correct nested path. Do not blindly apply the `best.ckpt` symlink trick — for these two runs the symlinks were never created; reference the nested paths directly (see `RUNS` in `scripts/eval_spatial_full.py`).

15. **`feat_chmean` collapses on the spatial backbone**: For the spatial models the channel-mean of pre-fusion features is negative-skewed, so `feat_chmean` collapses to a "predict everything as disease" failure mode (DisIoU ≈ 17.9%, BG IoU = 0%). Use `feat_chvar` instead — it's the recommended pre-fusion seed mode for spatial-trained checkpoints.

---

## 13. Siamese Network / ADPL-CAM Investigation (April 2026)

### 13.1 Paper Under Review

**Paper**: "Weakly supervised localization model for plant disease based on Siamese networks" — Chen J, Guo J, Zhang H, Liang Z, Wang S (2024). *Frontiers in Plant Science* 15:1418201. doi: 10.3389/fpls.2024.1418201

**PDF location**: `weakly_survised_localization.pdf` (repo root)

**GitHub**: https://AutoGo-Lab/SPDNet

### 13.2 Paper Summary: How SPDNet + ADPL-CAM Works

**Architecture (SPDNet)**:
- Siamese network with two weight-sharing branches processing image pairs (query + same-class reference)
- Each branch: MBConv blocks + Transformer blocks + Multi-Scale Excitation (MSE, SE-Net variant using GAP+GMP)
- Feature Pyramid Network (FPN) fuses multi-scale features from both branches
- PReLU activations, dropout=0.5

**Training task**: Multi-class disease classification (NOT similarity/contrastive learning). The network predicts disease class of the query image. The Siamese structure is a feature enhancement trick — the reference image's features are tokenized and fused into query features before classification. Loss is standard classification cross-entropy. Metrics: Top-K accuracy, precision, recall, F1.

**Pair formation**: For each query image, a random same-class image is sampled as the reference. Both are processed through the shared-weight branches.

**ADPL-CAM (deconstructed — despite "proprietary" label, fully described in Sections 3.2.1–3.2.4)**:

| Component | Description | Standard equivalent |
|-----------|-------------|-------------------|
| Multi-scale CAM fusion (Eq. 16) | Weighted sum of FPN feature maps: CAM = Σ w_i · F_i | Standard multi-scale CAM |
| Feature tokenization (Eq. 17) | GlobalMaxPool on reference features → compact tokens T_i per scale | Channel-wise max pooling (SE-style) |
| Token-based fusion (Eq. 18) | G' = G + Σ α·T_i — reference tokens modulate query features | Cross-image feature modulation (few-shot prototypes) |
| Adaptive thresholding (Eq. 19) | Bradley local threshold for CAM binarization | Standard Bradley thresholding |
| NMS (Sec. 3.2.4) | Non-maximum suppression to generate bounding boxes from CAM | Standard NMS |

**Key insight**: ADPL-CAM = multi-scale CAM + reference-guided feature modulation + standard post-processing. No fundamentally novel algorithm.

### 13.3 Paper Results

| Dataset | Backbone | Method | Top-1 Acc | Average IoU (bbox) |
|---------|----------|--------|-----------|-------------------|
| CUB-200 | ResNet50 | GradCAM | 46.71% | 40.3% |
| CUB-200 | ResNet50 | ADPL-CAM | 48.56% | 51.2% |
| CUB-200 | SPDNet | ADPL-CAM | 54.29% | 67.5% |
| MCPDD (plants) | SPDNet | ADPL-CAM | 62.88–97.09% | 42.04–63.25% |

**Critical caveats**:
- MCPDD dataset has only **42 images** total (grapes, potato, tomato under different lighting)
- Average IoU is **bounding-box IoU**, not pixel-level segmentation IoU
- CUB-200 Top-1 of 54.29% is well below SOTA (~90%+)
- No pixel-level mask evaluation anywhere in the paper

### 13.4 Relevance Assessment for Our WSSS Pipeline

**Why it's interesting**:
1. The reference-guided CAM enhancement principle is sound — forcing the network to focus on disease-discriminative features by comparing against a same-class reference image
2. Paper's CAM visualizations show spatially precise heat spots on individual disease lesions (much more focused than our binary MCTformer's diffuse activations)
3. Multi-scale FPN features could capture both small lesions and large infected areas (vs MCTformer's single-resolution ViT attention)
4. Designed specifically for the plant disease domain

**Why caution is needed**:
1. **Tiny evaluation** (42 images), cherry-picked visualizations
2. **Bounding-box-only evaluation** — no pixel-level IoU, which is what we need for WSSS
3. **Our MC115 approach already exploits the same principle** (fine-grained discrimination) via multi-class training — we saw +120% disease IoU improvement from binary → MC115
4. **Visually precise heat spots ≠ good segmentation** — our own point sampling analysis showed ~53% of CAM peak points miss GT disease even at q=0.99
5. **Reference image requirement at inference** — adds complexity and variance

### 13.5 Implementation Plan (What Was Done)

The original plan had 3 steps:
- **Step 0**: Visualize MC115 MCTformer CAMs → **DONE** (see Section 5.7 motivation). Result: MCTformer does not discriminate disease-specific features.
- **Step 1**: Lightweight prototype-modulated CAM test (no retraining) → **SKIPPED**. Proceeded directly to full implementation.
- **Step 2**: Full Siamese implementation → **DONE** (see Sections 5.7.1–5.7.6 for full results).
- **Step 3**: Hybrid ensemble of MCTformer + SPDNet CAMs → **NOT DONE** (available as future work, low cost since both CAM sets exist).

### 13.6 Integration Points

SPDNet replaces **Steps 1-2** of the WSSS pipeline (classifier training + CAM generation). Everything downstream (CRF, PSA, Random Walk, WeakCLIP) remains unchanged — it all operates on per-pixel CAM maps regardless of how they were generated.

**Implemented files** (full list in Section 3 repo structure):
- `src/wsss/spdnet/` — model, dataset, lightning module, cam generator (4 files)
- `src/conf/spdnet.py` — Hydra config dataclasses
- `src/train_spdnet.py` — training entry point
- `src/generate_spdnet_cams.py` — CAM generation entry point
- `tests/test_spdnet.py` — 46 unit tests (incl. spatial cross-attention)
- `scripts/run_spdnet_experiments.sh` — token fusion experiment sweep (6 runs)
- `scripts/run_spdnet_spatial_experiments.sh` — spatial cross-attention experiments (3 runs)
- `scripts/visualize_spdnet_activations.py` — 8-panel visualization grids
- `scripts/eval_spdnet_checkpoints.py` — automated checkpoint evaluation
- `scripts/overnight_eval_and_train.sh` — overnight eval+training pipeline

### 13.7 Decision Gate & Outcome

**MCTformer CAM visualization (Step 0)**: Visualized MC115 CAMs on 25 validation images with multiple activation types (binary-agg CAM, per-class top-1/top-2, GradCAM, CLS/patch token attention maps). **Conclusion: MCTformer does not discriminate distinctive disease features.** Hot spots occasionally match disease spots but this is not consistently confirmed across samples. No activation type reliably highlighted disease regions.

**SPDNet implementation & evaluation (Step 2)**: Full Siamese network implemented, trained (5 runs), and evaluated. Best SPDNet CAM quality (28.08% disease IoU from N=1 heavy) is **comparable but slightly below** MC115 MCTformer (~29.98% disease IoU). The Siamese reference-guidance approach did not deliver the localization improvement suggested by the paper's cherry-picked visualizations on a 42-image dataset.

**Overall conclusion**: Neither MCTformer attention-based CAMs nor SPDNet reference-guided ADPL-CAMs (token *or* spatial fusion) produce sufficiently precise disease localization on PlantSeg (~28-32% disease IoU after CRF). The CAM quality ceiling appears to be a property of the classification-to-localization paradigm itself, not a specific architecture limitation — even the spatial cross-attention variant (April 2026, Sections 5.9–5.10) under classification-only training failed to improve localization despite a non-trivial learned gate. Future directions should consider: (a) explicit spatial supervision for SPDNet (Section 5.11 sketch — contrastive / equivariance / self-distillation losses), (b) the hybrid ensemble (cheap, untested), (c) fundamentally different approaches to WSSS (e.g., self-supervised pretraining, text-guided segmentation, or foundation model adaptation).

### 13.8 Detailed SPDNet Implementation Notes (for Agents)

**Hydra config structure** (`src/conf/spdnet.py`):
```python
@dataclass
class SPDNetDataConfig:
    image_size: int = 448
    batch_size: int = 16
    num_workers: int = 4
    num_references: int = 1          # N same-class refs per query
    augmentation: str = "heavy"      # "heavy" | "light" | "minimal"

@dataclass
class SPDNetModelConfig:
    backbone: str = "resnet50"
    num_classes: int = 115
    fpn_channels: int = 256
    pretrained: bool = True

@dataclass
class SPDNetTrainerConfig:
    max_epochs: int = 80
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    gradient_clip_val: float = 1.0

@dataclass
class SPDNetConfig:
    data: SPDNetDataConfig = field(default_factory=SPDNetDataConfig)
    model: SPDNetModelConfig = field(default_factory=SPDNetModelConfig)
    trainer: SPDNetTrainerConfig = field(default_factory=SPDNetTrainerConfig)
    seed: int = 42
    experiment_name: str = "spdnet_plantseg"
    run_name: Optional[str] = None   # explicit MLflow run name
```

**Key implementation details**:
- `SPDNet.forward(query, reference, return_cam=False)` — `reference` is either a single `[B,3,H,W]` tensor or a `list[Tensor]` for multi-reference. When list, each ref is processed through backbone+FPN+MSE independently, tokens averaged before fusion.
- `_merge_and_fuse(q_fpn, r_fpn)` — helper that (1) normalizes and sums query FPN levels, (2) extracts tokens via GlobalMaxPool on ref FPN features, (3) fuses: `merged + alpha * sum(tokens)/4`
- `ADPLCam.forward(merged, classifier_weight, return_cam)` — when `return_cam=True`, produces `[B, C, H, W]` per-class CAMs via einsum of features × classifier weights, followed by `F.relu`
- `train_spdnet.py:build_train_transform(aug_name)` — dispatch function returning appropriate torchvision transform pipeline
- `cam_generator.py:generate_all_cams()` — uses fixed `SEED=42` + `random.seed(SEED)` for reproducible reference sampling during CAM generation
- `cam_generator.py:binary_aggregate` supports `"max"`, `"mean"`, and `"top_energy"` modes

**Checkpoint naming convention**: `epoch={epoch:02d}-val_mAP={val/mAP:.4f}.ckpt`. Best checkpoints are symlinked to `best.ckpt` to avoid Hydra `=` parsing issues.

**Output directory convention**: `outputs/{experiment_name}/{run_name}/checkpoints/` when `run_name` is set explicitly, otherwise `outputs/{experiment_name}/{timestamp}/checkpoints/`.

**MLflow logging**: Experiment name from `cfg.experiment_name`, run name from `cfg.run_name`. Tags logged: `num_references`, `augmentation`, `image_size`, `seed`. Metrics: `train/loss`, `train/mAP`, `val/loss`, `val/mAP` (per epoch), step-level `train/loss_step`.

### 13.9 Experiment Sweep Script Details

`scripts/run_spdnet_experiments.sh` defines 6 runs:

| Run # | `RUNS` ID | run_name | N refs | Aug | Batch | Accum | Est. time |
|-------|-----------|----------|--------|-----|-------|-------|-----------|
| 1 | 1 | spdnet_fix_n1_heavy | 1 | heavy | 16 | 2 | ~2.5h |
| 2 | 2 | spdnet_fix_n3_heavy | 3 | heavy | 12 | 2 | ~3.5h |
| 3 | 3 | spdnet_fix_n5_heavy | 5 | heavy | 10 | 4 | ~5.2h |
| 4 | 4 | spdnet_fix_n8_heavy | 8 | heavy | 8 | 4 | ~7.8h |
| 5 | 5 | spdnet_fix_n3_light | 3 | light | 12 | 2 | ~3.5h |
| 6 | 6 | spdnet_fix_n3_minimal | 3 | minimal | 12 | 2 | ~3.5h |

Usage: `RUNS="1 2" ./scripts/run_spdnet_experiments.sh` to run only specific configs. All runs use 80 epochs, `log_every_n_steps=200`, `trainer.gradient_clip_val=1.0`.

Time estimates are for RTX 5090 (24GB). Runs 1, 2, 5, 6 completed. Runs 3, 4 not executed.

### 13.10 Post-hoc Feature Seed Discovery (April 2026)

After completing all SPDNet training and ADPL-CAM evaluation, a critical analysis revealed that **raw backbone features contain better localization information than classifier-projected CAMs**.

**What was done** (no retraining):
1. Added `extract_merged_features()` to SPDNet model to expose intermediate feature maps
2. Implemented `generate_spdnet_seed()` with 3 modes: `feat_chmean` (channel-mean), `feat_chmax` (channel-max), `spatial_proto` (cosine similarity with reference prototype)
3. Exposed `srgb` parameter in DenseCRF (was hardcoded to VOC default of 13)
4. Created CRF parameter sweep script (`scripts/sweep_crf_params.py`) sweeping srgb × bg_threshold × scale_factor
5. Evaluated all seed types on 200 val images with threshold sweeps + CRF sweep

**Result**: feat_chmean + CRF(srgb=5, bg_thr=0.30) achieves **42.13% disease IoU** — a **+12.15pp improvement** over MCTformer baseline and **+14.05pp** over SPDNet ADPL-CAM. See Section 5.8 for full details.

**Why this matters**: The localization information *is* in the network — it's the classification projection that destroys it. This has implications for the entire WSSS pipeline: future work should explore feature-based seeds throughout (MCTformer ViT features, not just CLS token attention).

**Spatial prototype result**: Inference-only cosine similarity with reference prototype did NOT outperform channel-mean (17.60% vs 36.50%). This confirms the GlobalMaxPool tokenization in ADPL-CAM loses all spatial information, and spatial reference matching requires architectural changes (e.g., cross-attention) and retraining.
