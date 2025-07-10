skip_done=false

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
      --category)
        category="$2"
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
    python src/generation/resize_videos.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device python src/generation/predict_human.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device blenderproc run src/generation/raycast.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device python src/generation/track.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device python src/generation/pnp.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device python src/generation/estimate_depth.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device blenderproc run src/generation/optimize_depth.py --dataset $dataset --category "$category" --skip_done
    CUDA_VISIBLE_DEVICES=$device blenderproc run src/generation/prepare_4dhoi.py --dataset $dataset --category "$category" --skip_done
else
    python src/generation/resize_videos.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device python src/generation/predict_human.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device blenderproc run src/generation/raycast.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device python src/generation/track.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device python src/generation/pnp.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device python src/generation/estimate_depth.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device blenderproc run src/generation/optimize_depth.py --dataset $dataset --category "$category"
    CUDA_VISIBLE_DEVICES=$device blenderproc run src/generation/prepare_4dhoi.py --dataset $dataset --category "$category"
fi