export GLOG_logtostderr=1
export GLOG_v=2

model_dir=pretrained_models/20210815_unified_conformer_libtorch
./build/bin/websocket_server_main \
    --port 10086 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log