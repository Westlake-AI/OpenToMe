#!/bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

# Check if correct number of arguments is provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <image_path> <merge_num> <gpu> <model_name>"
    echo "Example: $0 ./demo 100 1 deit_small_patch16_224"
    exit 1
fi

# Parse arguments
image_path=$1
merge_num=$2
gpu=$3
model_name=$4

# Define all ToMe methods to test
# tome_methods=("crossget" "dct" "dtem" "mctf" "pitome" "tofu" "tome")
tome_methods=("dct" "dtem" "mctf" "pitome" "tofu" "tome")

echo "Starting ToMe methods testing..."
echo "Image path: $image_path"
echo "Merge number: $merge_num"
echo "GPU: $gpu"
echo "Model: $model_name"
echo "Methods to test: ${tome_methods[*]}"
echo "=========================================="

# Loop through each ToMe method
for tome_method in "${tome_methods[@]}"; do
    echo ""
    echo "Testing method: $tome_method"
    echo "------------------------------------------"
    
    # Run the visualization script for current method
    if bash evaluations/visualizations/vis_eval.sh "$image_path" "$tome_method" "$merge_num" "$gpu" "$model_name"; then
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
