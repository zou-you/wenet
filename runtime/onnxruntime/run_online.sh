export GLOG_logtostderr=1
export GLOG_v=2

context_path=context.txt
model_dir=../pretrained_models/multi_cn_unified_conformer_exp

onnx_dir=$model_dir/onnx
units=$model_dir/units.txt  # Change it to your model units path


./build/bin/websocket_server_main \
    --port 10086 \
    --chunk_size 16 \
    --onnx_dir $onnx_dir \
    --unit_path $units 2>&1 | tee log.txt

    # --context_path $context_path \