. ./path.sh || exit 1;

dir=exp/finetune_粤语0

python wenet/bin/export_jit.py \
  --config $dir/train.yaml \
  --checkpoint $dir/45.pt \
  --output_file $dir/final.zip