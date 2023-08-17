export GLOG_logtostderr=1
export GLOG_v=2
wav_path=$1
use_contest=$2
language=$3

context_path=context.txt

if [ $language = 'MIX' ]; then
	model_dir=../pretrained_models/20210815_unified_conformer_exp
else
	model_dir=../pretrained_models/20210815_unified_conformer_exp
fi

onnx_dir=$model_dir/onnx
units=$model_dir/units.txt  # Change it to your model units path
./build/bin/decoder_main \
    --chunk_size 16 \
    --wav_path $wav_path \
    --onnx_dir $onnx_dir \
    --context_path $context_path \
    --unit_path $units 2>&1 | tee log.txt

