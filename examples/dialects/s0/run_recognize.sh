#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;
# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0"

# The num of nodes
num_nodes=1
# The rank of current node
node_rank=0

# 方言
dialect=粤语
echo "当前方言：$dialect"

# 数据地址
dict=data/$dialect/dict/lang_char.txt

# 处理成shards格式的存储位置
shards_dir=./dialect_shards/$dialect

# 训练参数设置
train_config=conf/train_conformer.yaml
checkpoint=/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/exp/finetune_粤语0/45.pt
dir=exp/$dialect
echo "checkpoint: $checkpoint"

# 解码设置
# decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decode_modes=ctc_prefix_beam_search


# Specify decoding_chunk_size if it's a unified dynamic chunk trained model
# -1 for full chunk
decoding_chunk_size=
ctc_weight=0.5
reverse_weight=0.0

test_sets=test

for mode in ${decode_modes}; do
{
  base=$(basename $checkpoint)
  result_dir=$dir/test_${mode}_${base}
  mkdir -p $result_dir
  python wenet/bin/recognize.py --gpu 0 \
    --mode $mode \
    --config $dir/train.yaml \
    --data_type "shard" \
    --test_data data/$dialect/data.list \
    --checkpoint $checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dict \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file $result_dir/text \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  python tools/compute-wer.py --char=1 --v=1 \
    data/$dialect/test/text $result_dir/text > $result_dir/wer
}
done


