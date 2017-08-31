set -x 

export CUDA_VISIBLE_DEVICES="1,2,3" 
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })
n_gpus=${#gpu_list[@]}

#python -m pdb train_multi_gpus.py \
python train_multi_gpus.py \
  --n_gpus=$n_gpus \
  --with_spatial_transform=True \
  --batch_size=3 \
  --n_epochs=2000 \
  --val_ratio=0.5 \
  --test_ratio=0.0 \
  --validate_every=1 \
  --checkpoint_every=200 \
  --max_checkpoints=3 

