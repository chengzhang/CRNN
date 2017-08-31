set -x
export CUDA_VISIBLE_DEVICES="1" 
python train.py \
  --with_spatial_transform=True \
  --batch_size=1 \
  --n_epochs=300 \
  --val_ratio=0.5 \
  --test_ratio=0.0 \
  --validate_every=1 \
  --checkpoint_every=100 \
  --max_checkpoints=3 

