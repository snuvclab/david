skip_done=false
inference_epoch=2000
lora_weight=0.9

# Function to parse arguments and set values
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --device)
        device="$2"
        shift 2
        ;;
      --dataset)
        dataset="$2"
        shift 2
        ;;
      --text_prompt)
        text_prompt="$2"
        shift 2
        ;;
      --category)
        category="$2"
        shift 2
        ;;
      --max_seed)
        max_seed="$2"
        shift 2
        ;;
      --inference_epoch)
        inference_epoch="$2"
        shift 2
        ;;
      --lora_weight)
        lora_weight="$2"
        shift 2
        ;;
      --skip_done)
        skip_done=true
        shift 1
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done
}

# Call the function to parse command-line arguments
parse_args "$@"

if [ "$skip_done" = true ]; then
    for seed in $(seq 0 $max_seed); do
        CUDA_VISIBLE_DEVICES=$device python src/david/inference_mdm.py \
            --david_dataset $dataset \
            --category $category \
            --seed $seed \
            --num_samples 1 \
            --num_repetitions 1 \
            --lora_weight $lora_weight \
            --inference_epoch $inference_epoch \
            --text_prompt "$text_prompt" \
            --skip_done
    done
    CUDA_VISIBLE_DEVICES=$device python src/david/joint2smplx.py --dataset $dataset --category $category --skip_done
else
    for seed in $(seq 0 $max_seed); do
        CUDA_VISIBLE_DEVICES=$device python src/david/inference_mdm.py \
            --david_dataset $dataset \
            --category $category \
            --seed $seed \
            --num_samples 1 \
            --num_repetitions 1 \
            --lora_weight $lora_weight \
            --inference_epoch $inference_epoch \
            --text_prompt "$text_prompt"
    done
    # CUDA_VISIBLE_DEVICES=$device python src/david/joint2smplx.py --dataset $dataset --category $category
fi

