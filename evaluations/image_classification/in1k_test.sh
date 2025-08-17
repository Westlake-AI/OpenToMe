#!/bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

# Check if correct number of arguments is provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <cuda> <merge_num> <dataset> <gpus> <model_name>"
    echo "Example: $0 0 100 ./data/ImageNet/val_folder 1 deit_small_patch16_224"
    exit 1
fi

# Parse arguments
cuda=$1
merge_num=$2
dataset=$3
gpus=$4
model_name=$5

# Define all ToMe methods to test
# tome_methods=("crossget" "dct" "dtem" "mctf" "pitome" "tofu" "tome")
tome_methods=("dct" "dtem" "mctf" "pitome" "tofu" "tome")

echo "Starting ToMe methods testing for image classification..."
echo "CUDA: $cuda"
echo "Merge number: $merge_num"
echo "Dataset: $dataset"
echo "GPUs: $gpus"
echo "Model: $model_name"
echo "Methods to test: ${tome_methods[*]}"
echo "=========================================="

# Loop through each ToMe method
for tome_method in "${tome_methods[@]}"; do
    echo ""
    echo "Testing method: $tome_method"
    echo "------------------------------------------"
    
    # Run the image classification script for current method
    if bash evaluations/image_classification/in1k_eval.sh "$cuda" "$tome_method" "$merge_num" "$dataset" "$gpus" "$model_name"; then
        echo "✅ Test PASSED for $tome_method"
        echo "------------------------------------------"
    else
        echo "❌ Test FAILED for $tome_method"
        echo "Exiting due to test failure..."
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "🎉 All ToMe methods tested successfully!"
echo "✅ All tests PASSED"
echo "=========================================="
