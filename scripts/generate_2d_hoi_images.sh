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
    blenderproc run src/generation/render_objects.py --dataset $dataset --category "$category" --skip_done
    python src/generation/extract_canny.py --dataset $dataset --category "$category" --skip_done
    python src/generation/generate_images.py --dataset $dataset --category "$category" --device $device --skip_done
else
    blenderproc run src/generation/render_objects.py --dataset $dataset --category "$category"
    python src/generation/extract_canny.py --dataset $dataset --category "$category"
    python src/generation/generate_images.py --dataset $dataset --category "$category" --device $device
fi