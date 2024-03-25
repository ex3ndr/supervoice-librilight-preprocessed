set -e

# echo "Combining librilight"
# python ./scripts/combine.py --dataset librilight
# tar -cf ./datasets/librilight-processed.tar -C ./datasets librilight-processed

echo "Combining librilight-medium"
python ./scripts/combine.py --dataset librilight-medium
tar -cf ./datasets/librilight-medium-processed.tar -C ./datasets librilight-medium-processed
