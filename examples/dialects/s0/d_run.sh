#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="3,4,5"
stage=0
stop_stage=5

# The num of nodes
num_nodes=1
# The rank of current node
node_rank=0

# 方言
dialect=粤语
echo "当前方言：$dialect"

# 数据地址
dialect_data_dir=/home/zouyou/workspaces/ASR/wenet/examples/dialects/s0/metadata/$dialect

# 处理成shards格式的存储位置
shards_dir=./dialect_shards/$dialect

# 数据集划分设置
train_set=train
dev_set=dev
test_sets=test

# 训练参数设置
train_config=conf/train_conformer.yaml
checkpoint=
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


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation"
  local/dialect_data_prep.sh \
    $dialect_data_dir \
    data/$dialect \
    $dialect || exit 1;
fi

dict=data/$dialect/dict/lang_char.txt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Make a dictionary"
    echo "dictionary: ${dict}"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    echo "▁ 2" >> ${dict} # ▁ is for space
    tools/text2token.py -s 1 -n 1 --space "▁" data/$dialect/${train_set}/text \
        | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' \
        | grep -v "▁" \
        | awk '{print $0 " " NR+2}' >> ${dict} \
        || exit 1;
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN
  if $cmvn; then
    full_size=`cat data/$dialect/${train_set}/wav.scp | wc -l`
    sampling_size=$((full_size / cmvn_sampling_divisor))
    shuf -n $sampling_size data/$dialect/$train_set/wav.scp \
      > data/$dialect/$train_set/wav.scp.sampled
    python3 tools/compute_cmvn_stats.py \
    --num_workers 16 \
    --train_config $train_config \
    --in_scp data/$dialect/$train_set/wav.scp.sampled \
    --out_cmvn data/$dialect/$train_set/global_cmvn \
    || exit 1;
  fi
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making shards, please wait..."
  RED='\033[0;31m'
  NOCOLOR='\033[0m'

  for x in $dev_set ${train_set}; do
  # for x in ${train_set}; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 1000 \
      --num_threads 32 --segments data/$dialect/$x/segments \
      data/$dialect/$x/wav.scp data/$dialect/$x/text \
      $(realpath $dst) data/$dialect/$x/data.list

  tools/make_raw_list.py --segments data/$dialect/$test_sets/segments data/$dialect/$test_sets/wav.scp \
    data/$dialect/$test_sets/text data/$dialect/$test_sets/data.list

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
  $cmvn && cp data/$dialect/${train_set}/global_cmvn $dir
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
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  for testset in ${test_sets} ${dev_set}; do
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
