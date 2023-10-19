#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="1,2,3,4,5"
stage=0
stop_stage=5

# The num of nodes
num_nodes=1
# The rank of current node
node_rank=0

# 方言
dialect=finetune_四川话
echo "当前方言：$dialect"


# 处理成shards格式的存储位置
shards_dir=./dialect_shards/$dialect

# 数据集划分设置
train_set=train
dev_set=dev
test_sets=test

# 训练参数设置
train_config=conf/train_finetune_sichuan.yaml
checkpoint=/home/zouyou/workspaces/ASR/wenet/examples/wenetspeech/s0/pretrained_models/20220506_u2pp_conformer_exp/final.pt
cmvn=true
cmvn_sampling_divisor=20 # 20 means 5% of the training data to estimate cmvn
dir=exp/$dialect
echo "checkpoint: $checkpoint"

# 解码设置
decode_checkpoint=
average_checkpoint=true
average_num=30
# decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decode_modes=ctc_prefix_beam_search

. tools/parse_options.sh || exit 1;

set -u
set -o pipefail

dict=data/$dialect/dict/lang_char.txt


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making shards, please wait..."
  RED='\033[0;31m'
  NOCOLOR='\033[0m'

  for x in $dev_set $test_sets ${train_set}; do
  # for x in ${train_set}; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 1000 \
      --num_threads 32 --segments data/$dialect/$x/segments \
      data/$dialect/$x/wav.scp data/$dialect/$x/text \
      $(realpath $dst) data/$dialect/$x/data.list

  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training"
  mkdir -p $dir
  # INIT_FILE is for DDP synchronization
  INIT_FILE=$dir/ddp_init
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp /home/zouyou/workspaces/ASR/wenet/examples/wenetspeech/s0/pretrained_models/20220506_u2pp_conformer_exp/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type "shard" \
      --symbol_table $dict \
      --train_data data/$dialect/$train_set/data.list \
      --cv_data data/$dialect/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      $cmvn_opts \
      --num_workers 8 \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --val_best
  fi
  echo "save finished"
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  for testset in ${test_sets}; do
  {
    for mode in ${decode_modes}; do
    {
      base=$(basename $decode_checkpoint)
      result_dir=$dir/${testset}_${mode}_${base}
      mkdir -p $result_dir
      python wenet/bin/recognize.py --gpu 0 \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type "shard" \
        --test_data data/$dialect/$testset/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $dict \
        --ctc_weight $ctc_weight \
        --reverse_weight $reverse_weight \
        --result_file $result_dir/text \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
      python tools/compute-wer.py --char=1 --v=1 \
        data/$dialect/$testset/text $result_dir/text > $result_dir/wer
    }
    done
    wait
  }
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Export the best model you want"
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg${average_num}.pt \
    --output_file $dir/final.zip
fi
