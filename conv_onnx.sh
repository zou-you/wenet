exp=/home/zouyou/workspaces/ASR/wenet/runtime/pretrained_models/multi_cn_unified_conformer_exp # Change it to your experiment dir
onnx_dir=$exp/onnx
python -m wenet.bin.export_onnx_cpu \
  --config $exp/train.yaml \
  --checkpoint $exp/final.pt \
  --chunk_size 16 \
  --output_dir $onnx_dir \
  --num_decoding_left_chunks -1

# When it finishes, you can find `encoder.onnx`, `ctc.onnx`, and `decoder.onnx` in the $onnx_dir respectively.