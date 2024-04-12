set -e
python ./scripts/cut_by_vad.py --input_dir "./external_datasets/librilight" --output_dir "./datasets/librilight" --target_len_sec 15 --hard_target_sec 30
python ./scripts/cut_by_vad.py --input_dir "./external_datasets/librilight-medium" --output_dir "./datasets/librilight-medium" --target_len_sec 15 --hard_target_sec 30
python ./scripts/cut_by_vad.py --input_dir "./external_datasets/librilight-large" --output_dir "./datasets/librilight-large" --target_len_sec 15 --hard_target_sec 30