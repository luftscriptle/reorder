import argparse
import cv2
import os
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video")
    parser.add_argument("--noise_folder", default="/home/lmalartic/Desktop/code_tmp/technical/noise")
    return parser   


def main():
    args = make_parser().parse_args()
    # open all frames in noise folder

    noise_frames = []
    for file in os.listdir(args.noise_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            noise_frames.append(os.path.join(args.noise_folder, file))
    print(f"Found {len(noise_frames)} noise frames.")
    # Load noise frames
    loaded_noise_frames = [cv2.imread(frame) for frame in noise_frames]
    # Open video and get all frames

    cap = cv2.VideoCapture(args.source_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")
    frames = loaded_noise_frames.copy()
    success, frame = cap.read()
    max_frames = 100
    while success and len(frames) < max_frames:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    print(f"Extracted {len(frames)} frames from video.")
    # Shuffle frames and save to mp4
    # resize all frames to the size of the first frame
    height, width, layers = frames[0].shape
    frames = [cv2.resize(f, (width, height)) for f in frames]
    np.random.seed(42)
    indices = np.arange(len(frames))
    np.random.shuffle(indices)
    shuffled_frames = [frames[i] for i in indices]
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('shuffled_video.mp4', fourcc, 30, (width, height))
    print(f"Saving shuffled video with {len(shuffled_frames)} frames.")
    for frame in shuffled_frames:
        out.write(frame)
    out.release()
    print("Saved shuffled video to shuffled_video.mp4")

if __name__ == '__main__':
    main()
