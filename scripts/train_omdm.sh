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
    CUDA_VISIBLE_DEVICES=$device python src/david/process_omdm.py --dataset $dataset --category $category
    CUDA_VISIBLE_DEVICES=$device python src/david/train_omdm.py --dataset $dataset --category $category
else
    CUDA_VISIBLE_DEVICES=$device python src/david/process_omdm.py --dataset $dataset --category $category
    CUDA_VISIBLE_DEVICES=$device python src/david/train_omdm.py --dataset $dataset --category $category
fi

