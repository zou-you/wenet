export GLOG_logtostderr=1
export GLOG_v=2
wav_path=$1
use_contest=$2
language=$3

context_path=context.txt

if [ $language = 'MIX' ]; then
	model_dir=pretrained_models/20210815_unified_conformer_libtorch
else
	model_dir=pretrained_models/20220506_u2pp_conformer_libtorch
fi


echo $use_contest

if [ $use_contest = 'Y' ]; then
    echo 'use contest'
    ./build/bin/decoder_main \
        --chunk_size 16 \
        --wav_path $wav_path \
        --model_path $model_dir/final.zip \
        --context_path $context_path \
        --context_score 6 \
        --unit_path $model_dir/units.txt 2>&1 | tee log.txt
else
    echo 'not use contest'
    ./build/bin/decoder_main \
        --chunk_size 16 \
        --wav_path $wav_path \
        --model_path $model_dir/final.zip \
        --unit_path $model_dir/units.txt 2>&1 | tee log.txt
fi
