#!/usr/bin/env bash
set -euo pipefail

INPUT_CONFIG="/home/lmalartic/Desktop/code_tmp/technical/configs/full_config.yaml"

# -c flag to override config
while getopts c: flag; do
  case "${flag}" in
    c) INPUT_CONFIG=${OPTARG};;
  esac
done

echo "Using config file: $INPUT_CONFIG"

now=$(date +"%Y_%m_%d-%H_%M_%S")

# Read values (Python yq)
VIDEO_FILE=$(yq -r '.video.input_path' "$INPUT_CONFIG")
OUTPUT_DIR=$(yq -r '.main.output_dir' "$INPUT_CONFIG")


new_output_dir="${OUTPUT_DIR}/run_${now}"

# Create output dir
mkdir -p "$new_output_dir"

cp $INPUT_CONFIG "${new_output_dir}/config.yaml" 
CONFIG_FILE="${new_output_dir}/config.yaml"
NEW_OUTPUT_DIR="$new_output_dir" yq -y '.main.output_dir = env.NEW_OUTPUT_DIR' "$INPUT_CONFIG" > "$CONFIG_FILE"


mkdir -p "${new_output_dir}/frames"
FRAMES_DIR="${new_output_dir}/frames"
tmp="$(mktemp)"

FRAMES_DIR="$FRAMES_DIR" \
yq -y '.video.frames_dir = env.FRAMES_DIR' "$CONFIG_FILE" > "$tmp" \
  && mv "$tmp" "$CONFIG_FILE"
echo "Extracting frames from: $VIDEO_FILE"
echo "Output directory: ${new_output_dir}/frames"
ffmpeg -hide_banner -loglevel error -stats \
  -i "$VIDEO_FILE" -qscale:v 2 "${new_output_dir}/frames/raw_frames_%04d.jpg"
FRAME_COUNT=$(ls -1 "${new_output_dir}/frames" | wc -l)
echo "Successfully extracted $FRAME_COUNT frames to ${new_output_dir}/frames"
# Display the config 

echo "Using config file: $CONFIG_FILE"
# Compute embeddings and visualize
mkdir -p "${new_output_dir}/embeddings"
mkdir -p "${new_output_dir}/clustering"
mkdir -p "${new_output_dir}/visualizations"

python src/compute_embeddings.py --config $CONFIG_FILE

python src/cluster_embeddings.py --config $CONFIG_FILE
# new_output_dir="/home/lmalartic/Desktop/code_tmp/technical/outputs/run_2025_10_06-13_02_14"
for f in "${new_output_dir}/clustering"/cluster_*; do
    echo "Processing folder: $f"    
    if [ -d "$f" ]; then
        echo "Cluster folder: $f"
        python src/reorder_frames.py --frames "$f"
        # Use ffmpeg to produce a video from the reordered frames
        ffmpeg -hide_banner -loglevel error -stats -framerate 30 -i "$f/reordered_frames/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p "$f/reordered_video.mp4"
        echo "Reordered video saved to: $f/reordered_video.mp4"
    else
        echo "$f is not a directory, skipping."
    fi
done
