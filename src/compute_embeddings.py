#!/usr/bin/env python3

import argparse
import subprocess
import os
import yaml
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
from PIL import Image
import clip
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader


class FrameDataset(Dataset):
    def __init__(self, frames_dir, extensions, preprocess):
        self.frames_dir = frames_dir
        self.preprocess = preprocess
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.frames_dir, f"*{ext}")))
        self.image_paths = sorted(self.image_paths)
        
        if not self.image_paths:
            raise ValueError(f"No images found in {frames_dir} with extensions {extensions}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        return image, os.path.basename(image_path)


def extract_frames(config):
    video_file = config['video']['input_path']
    output_dir = config['video']['frames_dir']
    fps = config['video'].get('fps', '')
    format = config['video'].get('format', 'jpg')
    
    print(f"Extracting frames from {video_file}...")
    
    cmd = ['bash', 'split_video.sh', video_file, format]
    if fps:
        cmd.append(str(fps))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting frames: {result.stderr}")
        return False
    
    print(f"Frames extracted to {output_dir}")
    return True


def compute_embeddings(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {config['embedding']['model_name']}")
    model, preprocess = clip.load(config['embedding']['model_name'], device=device)
    model.eval()
    
    frames_dir = config['video']['frames_dir']

    print(f"Loading frames from: {frames_dir}")
    dataset = FrameDataset(
        frames_dir,
        config['video']['image_extensions'],
        preprocess
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['embedding']['batch_size'],
        num_workers=config['embedding']['num_workers'],
        shuffle=False,
        pin_memory=(device == "cuda")
    )
    
    embeddings_list = []
    filenames_list = []
    
    print(f"Computing embeddings for {len(dataset)} frames...")
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader), desc="Processing batches")
        for images, filenames in dataloader:
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings_list.append(image_features.cpu().numpy())
            filenames_list.extend(filenames)
            pbar.update(images.size(0))
    
    embeddings = np.vstack(embeddings_list)
    return embeddings, filenames_list


def reduce_dimensions(embeddings, method='tsne', n_components=2):
    print(f"Reducing dimensions using {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: {explained_var[:2] * 100}%")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=min(30, len(embeddings)-1), 
                      random_state=42, max_iter=1000)
        reduced = reducer.fit_transform(embeddings)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, n_neighbors=min(15, len(embeddings)-1),
                          min_dist=0.1, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reduced


def create_visualization(reduced_embeddings, filenames, method='tsne', output_file=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    
    scatter = ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=100, alpha=0.7)
    
    for i, (xi, yi, fname) in enumerate(zip(x, y, filenames)):
        frame_num = fname.replace('frame_', '').replace('.jpg', '').replace('.png', '')
        ax.annotate(frame_num, (xi, yi), fontsize=8, ha='center', va='bottom')
    
    plt.colorbar(scatter, ax=ax, label='Frame sequence')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'CLIP Embeddings Visualization ({method.upper()})\n')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved {method.upper()} visualization to: {output_file}")
    
    return fig


def visualize_embeddings(embeddings, filenames, config):
    methods = config['visualization']['methods']
    output_dir = config['main']['output_dir']

    for method in methods:
        reduced = reduce_dimensions(embeddings, method=method)
        output_file = os.path.join(output_dir, "visualizations", f"{method}_visualization.png")
        create_visualization(reduced, filenames, method=method, output_file=output_file)
        print(f"Shape: {embeddings.shape} -> {reduced.shape}")
        print(f"Min/Max values: [{reduced.min():.2f}, {reduced.max():.2f}]")


def save_embeddings(embeddings, filenames, output_file):
    import json
    data = {
        'filenames': filenames,
        'embeddings': embeddings.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved embeddings to: {output_file}")

def run_pipeline(config):
    embeddings, filenames = compute_embeddings(config)
    
    visualize_embeddings(embeddings, filenames, config)
    save_embeddings(embeddings, filenames, os.path.join(config['main']['output_dir'], "embeddings", "frame_embeddings.json"))
    
    print("\nPipeline completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description='End-to-end CLIP embeddings pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_pipeline(config)


if __name__ == "__main__":
    main()