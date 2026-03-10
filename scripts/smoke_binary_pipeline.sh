#!/bin/bash
set -e

# Smoke test: binary WSSS pipeline on 25-image PlantSeg subset
#
# Uses real PlantSeg + PlantVillage for MCTformer training (2 epochs),
# then generates CAMs and runs full refinement on 25 PlantSeg images.
#
# Expected runtime: ~10-15 min on GPU.
#
# Usage:
#   ./scripts/smoke_binary_pipeline.sh

export PATH="/venv/main/bin:$PATH"
cd /workspace/plant-diseases-segmentation

SMOKE_DIR="outputs/smoke_binary"
DATA_ROOT="data/plantsegv3"
PV_ROOT="data/plant-village"

rm -rf "${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}"

echo "============================================"
echo "  Binary Pipeline Smoke Test (25 images)"
echo "============================================"
echo ""

# ─── Create a 25-image PlantSeg subset for CAM generation ────
echo "=== Setup: sample 25 PlantSeg images ==="
MINI_PS="${SMOKE_DIR}/plantseg_mini"
mkdir -p "${MINI_PS}/images/train" "${MINI_PS}/annotations/train"
mkdir -p "${MINI_PS}/images/val" "${MINI_PS}/annotations/val"

python -c "
import random, shutil, json
from pathlib import Path

random.seed(42)
ps_img = Path('${DATA_ROOT}/images/train')
ps_ann = Path('${DATA_ROOT}/annotations/train')
mini_img = Path('${MINI_PS}/images/train')
mini_ann = Path('${MINI_PS}/annotations/train')

images = sorted(ps_img.glob('*.jpg'))
sample = random.sample(images, min(25, len(images)))
for img in sample:
    shutil.copy(img, mini_img / img.name)
    ann = ps_ann / f'{img.stem}.png'
    if ann.exists():
        shutil.copy(ann, mini_ann / ann.name)
print(f'Sampled {len(sample)} PlantSeg images')

# Copy a few to val too
for img in sample[:5]:
    shutil.copy(mini_img / img.name, Path('${MINI_PS}/images/val') / img.name)
    ann = mini_ann / f'{img.stem}.png'
    if ann.exists():
        shutil.copy(ann, Path('${MINI_PS}/annotations/val') / ann.name)
"

# ─── Pipeline config ─────────────────────────────────────────
OUT="${SMOKE_DIR}/pipeline"
NUM_FG=1
NUM_CLS=2
LABEL_FILE="${OUT}/labels/plantseg_binary_train.npy"
CLASS_NAMES="${OUT}/labels/class_names.txt"
BINARY_GT="${OUT}/gt_binary_train"
BINARY_GT_VAL="${OUT}/gt_binary_val"

# ─── Step 0: Export PlantSeg-only labels (for CAMs) ──────────
echo ""
echo "=== Step 0: Export labels ==="
python src/export_labels.py \
    mode=plantseg_binary \
    root="${MINI_PS}" \
    pv_split=train \
    include_plantvillage=false \
    output="${LABEL_FILE}"

# ─── Step 0c: Binary GT masks ────────────────────────────────
echo ""
echo "=== Step 0c: Generate binary GT masks ==="
python -c "
from pathlib import Path; import numpy as np; from PIL import Image
for src_dir, dst_dir in [
    ('${MINI_PS}/annotations/train', '${BINARY_GT}'),
    ('${MINI_PS}/annotations/val', '${BINARY_GT_VAL}'),
]:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src.glob('*.png')):
        m = np.array(Image.open(f))
        m[(m > 0) & (m < 255)] = 1
        Image.fromarray(m.astype(np.uint8)).save(dst / f.name)
        count += 1
    print(f'  {src_dir} -> {count} binary GT masks')
"

# ─── Step 1: Train MCTformer (2 epochs, binary, real data) ───
echo ""
echo "=== Step 1: Train binary MCTformer (2 epochs) ==="
python src/train_mctformer.py \
    dataset=plantseg_binary \
    experiment_name="smoke_mctformer_binary" \
    seed=0 \
    model.name=mctformer_v2 \
    model.pretrained=true \
    model.input_size=224 \
    plantseg_data.root="${MINI_PS}" \
    plantseg_data.pv_root="${PV_ROOT}" \
    plantseg_data.image_size=224 \
    plantseg_data.batch_size=8 \
    plantseg_data.num_workers=4 \
    trainer.max_epochs=2 \
    trainer.precision="32" \
    output_dir="${OUT}/mctformer"

CKPT=$(ls -t "${OUT}/mctformer/checkpoints/last.ckpt" 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: No MCTformer checkpoint"; exit 1
fi
echo "Checkpoint: ${CKPT}"

# ─── Step 2: Generate CAMs (25 PlantSeg images) ─────────────
echo ""
echo "=== Step 2: Generate CAMs ==="
CAM_DIR="${OUT}/cams/cam_npy"
python src/generate_cams.py \
    "checkpoint='${CKPT}'" \
    image_dir="${MINI_PS}/images/train" \
    image_ext=".jpg" \
    "label_file=${LABEL_FILE}" \
    output_dir="${CAM_DIR}" \
    num_classes=${NUM_FG} \
    input_size=224 \
    max_size=448 \
    "scales=[1.0]" \
    n_layers=3 \
    attention_type=fused \
    patch_attn_refine=true \
    gt_dir="${BINARY_GT}" \
    eval_threshold_sweep=true \
    eval_sweep_samples=25

echo "--- Verify CAM files ---"
python -c "
import numpy as np; from pathlib import Path
cam_dir = Path('${CAM_DIR}')
files = list(cam_dir.glob('*.npy'))
print(f'  CAM files: {len(files)}')
assert len(files) > 0, 'No CAM files!'
for f in files[:5]:
    d = np.load(str(f), allow_pickle=True).item()
    assert set(d.keys()) == {0}, f'Expected key {{0}}, got {set(d.keys())}'
    assert d[0].min() >= 0 and d[0].max() <= 1, f'Values out of [0,1]'
print('  OK: single key {0}, values in [0,1]')
"

# ─── Step 3: CRF ────────────────────────────────────────────
echo ""
echo "=== Step 3: Apply CRF ==="
LA_CRF="${OUT}/cams/la_crf"
HA_CRF="${OUT}/cams/ha_crf"
python src/apply_crf.py \
    cam_dir="${CAM_DIR}" \
    image_dir="${MINI_PS}/images/train" \
    image_ext=".jpg" \
    la_crf_dir="${LA_CRF}" \
    ha_crf_dir="${HA_CRF}" \
    bg_threshold=0.3 \
    la_scale_factor=1.0 \
    ha_scale_factor=12.0 \
    crf_iters=10 \
    num_cls=${NUM_CLS} \
    num_workers=4

echo "--- Verify CRF masks ---"
python -c "
import numpy as np; from pathlib import Path
for d in ['${LA_CRF}', '${HA_CRF}']:
    files = list(Path(d).glob('*.npy'))
    print(f'  {Path(d).name}: {len(files)} masks')
    assert len(files) > 0, f'No masks in {d}'
    m = np.load(str(files[0]))
    vals = set(np.unique(m))
    assert vals.issubset({0, 1, 255}), f'Unexpected values: {vals}'
print('  OK: CRF masks have values in {0, 1, 255}')
"

# ─── Step 4: Evaluate CRF ───────────────────────────────────
echo ""
echo "=== Step 4: Evaluate CRF masks ==="
python src/evaluate_masks.py \
    pred_dir="${LA_CRF}" \
    gt_dir="${BINARY_GT}" \
    num_cls=${NUM_CLS} \
    class_names_file="${CLASS_NAMES}"

# ─── Step 5: Train PSA (2 epochs) ───────────────────────────
echo ""
echo "=== Step 5: Train PSA (2 epochs) ==="
PSA_CKPT="${OUT}/psa/psa_aff.pth"
python src/train_psa.py \
    image_dir="${MINI_PS}/images/train" \
    image_ext=".jpg" \
    la_crf_dir="${LA_CRF}" \
    ha_crf_dir="${HA_CRF}" \
    backbone_weights="pretrained/res38_cls.pth" \
    output_path="${PSA_CKPT}" \
    batch_size=4 \
    max_epochs=2 \
    lr=0.01 \
    num_workers=4 \
    cropsize=224

# ─── Step 6: Random Walk ────────────────────────────────────
echo ""
echo "=== Step 6: Random Walk ==="
PSEUDO="${OUT}/pseudo_masks"
python src/run_random_walk.py \
    cam_dir="${CAM_DIR}" \
    image_dir="${MINI_PS}/images/train" \
    image_ext=".jpg" \
    aff_checkpoint="${PSA_CKPT}" \
    output_dir="${PSEUDO}" \
    bg_threshold=0.39 \
    beta=8 \
    logt=6 \
    num_cls=${NUM_CLS} \
    cropsize=224 \
    max_size=448

echo "--- Verify pseudo masks ---"
python -c "
import numpy as np; from pathlib import Path; from PIL import Image
files = list(Path('${PSEUDO}').glob('*.png'))
print(f'  Pseudo masks: {len(files)}')
assert len(files) > 0, 'No pseudo masks!'
for f in files[:5]:
    m = np.array(Image.open(f))
    assert set(np.unique(m)).issubset({0, 1, 255}), f'Bad values in {f.name}: {set(np.unique(m))}'
print('  OK: values in {0, 1, 255}')
"

# ─── Step 7: Evaluate pseudo masks ──────────────────────────
echo ""
echo "=== Step 7: Evaluate pseudo masks ==="
python src/evaluate_masks.py \
    pred_dir="${PSEUDO}" \
    gt_dir="${BINARY_GT}" \
    num_cls=${NUM_CLS} \
    class_names_file="${CLASS_NAMES}"

echo ""
echo "============================================"
echo "  Smoke test PASSED!"
echo "  MCTformer trained on PlantSeg+PlantVillage (binary, 2 epochs)"
echo "  CAMs + CRF + PSA + RW on 25 PlantSeg images"
echo "  All intermediate formats verified."
echo "  WeakCLIP steps skipped (requires ViT-B-16.pt)"
echo "============================================"
