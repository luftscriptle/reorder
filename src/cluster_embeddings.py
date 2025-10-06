#!/usr/bin/env python3

import argparse
import json
import yaml
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import os
import shutil


def load_config(config_file='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_embeddings_from_json(json_file):
    print(f"Loading embeddings from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    filenames = data['filenames']
    embeddings = data['embeddings']
    frame_numbers = [int(os.path.splitext(fn)[0].removeprefix("raw_frames_")) for fn in filenames]
    embeddings = np.array(embeddings)
    
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return frame_numbers, embeddings


def perform_dbscan_clustering(embeddings, eps=0.3, min_samples=3, metric='cosine'):
    print(f"\nPerforming DBSCAN clustering:")
    print(f"  - eps (max distance): {eps}")
    print(f"  - min_samples: {min_samples}")
    print(f"  - metric: {metric}")
    
    if metric == 'cosine':
        normalized_embeddings = normalize(embeddings, norm='l2')
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = dbscan.fit_predict(normalized_embeddings)
    else:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        clusters = dbscan.fit_predict(embeddings)
    
    return clusters


def analyze_clusters(frame_numbers, clusters):
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = len(clusters[clusters == -1])
    
    print(f"\nClustering Results:")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Number of noise points: {n_noise}")
    print(f"  - Total frames: {len(frame_numbers)}")
    
    cluster_info = {}
    
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        frames_in_cluster = [frame_numbers[i] for i in range(len(frame_numbers)) if mask[i]]
        
        if cluster_id == -1:
            cluster_info['noise'] = {
                'frames': frames_in_cluster,
                'count': len(frames_in_cluster),
                'percentage': len(frames_in_cluster) / len(frame_numbers) * 100
            }
        else:
            cluster_info[f'cluster_{cluster_id}'] = {
                'frames': frames_in_cluster,
                'count': len(frames_in_cluster),
                'percentage': len(frames_in_cluster) / len(frame_numbers) * 100,
                'frame_range': f"{min(frames_in_cluster)}-{max(frames_in_cluster)}"
            }
    
    print("\nCluster Details:")
    for cluster_name, info in sorted(cluster_info.items()):
        if cluster_name == 'noise':
            print(f"  Noise: {info['count']} frames ({info['percentage']:.1f}%)")
        else:
            print(f"  {cluster_name}: {info['count']} frames ({info['percentage']:.1f}%), range: {info['frame_range']}")
    
    return cluster_info


def save_frames_by_cluster(frame_numbers, clusters, config):
    """Copy frames to cluster directories"""
    frames_dir = os.path.join(config["main"]["output_dir"], "frames")
    clustering_dir = os.path.join(config["main"]["output_dir"], "clustering")
    
    # Create clustering directory if it doesn't exist
    os.makedirs(clustering_dir, exist_ok=True)
    
    unique_clusters = np.unique(clusters)
    
    print(f"\nSaving frames to cluster directories:")
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_dir = os.path.join(clustering_dir, "noise")
        else:
            cluster_dir = os.path.join(clustering_dir, f"cluster_{cluster_id}")
        
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Find frames belonging to this cluster
        mask = clusters == cluster_id
        cluster_frames = [frame_numbers[i] for i in range(len(frame_numbers)) if mask[i]]
        
        for frame_num in cluster_frames:
            source_file = os.path.join(frames_dir, f"raw_frames_{frame_num:04d}.jpg")
            dest_file = os.path.join(cluster_dir, f"raw_frames_{frame_num:04d}.jpg")

            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
            else:
                raise FileNotFoundError(f"Source file not found: {source_file}")
        
        if cluster_id == -1:
            print(f"  - Copied {len(cluster_frames)} frames to noise directory")
        else:
            print(f"  - Copied {len(cluster_frames)} frames to cluster_{cluster_id}")


def visualize_clusters(frame_numbers, clusters, output_file='clusters_visualization.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}
    color_map[-1] = [0.5, 0.5, 0.5, 1.0]
    
    for i, (frame, cluster) in enumerate(zip(frame_numbers, clusters)):
        color = color_map[cluster]
        ax1.bar(i, 1, color=color, width=1.0)
    
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Cluster')
    ax1.set_title('Temporal Cluster Distribution')
    ax1.set_xlim(0, len(frame_numbers))
    
    cluster_counts = Counter(clusters)
    cluster_labels = []
    cluster_sizes = []
    cluster_colors = []
    
    for cluster_id in sorted(cluster_counts.keys()):
        if cluster_id == -1:
            cluster_labels.append('Noise')
        else:
            cluster_labels.append(f'Cluster {cluster_id}')
        cluster_sizes.append(cluster_counts[cluster_id])
        cluster_colors.append(color_map[cluster_id])
    
    ax2.bar(cluster_labels, cluster_sizes, color=cluster_colors)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Frames')
    ax2.set_title('Cluster Size Distribution')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")


def compute_similarity_matrix(embeddings, sample_size=None):
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        indices = np.sort(indices)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
        indices = np.arange(len(embeddings))
    
    normalized = normalize(sample_embeddings, norm='l2')
    similarity_matrix = cosine_similarity(normalized)
    
    return similarity_matrix, indices


def visualize_similarity_matrix(embeddings, frame_numbers, output_file='similarity_matrix.png', sample_size=100):
    print(f"\nComputing similarity matrix...")
    similarity_matrix, indices = compute_similarity_matrix(embeddings, sample_size)
    sampled_frames = [frame_numbers[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_title('Cosine Similarity Matrix')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Frame Number')
    
    n_ticks = min(10, len(sampled_frames))
    tick_indices = np.linspace(0, len(sampled_frames) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([str(sampled_frames[i]) for i in tick_indices], rotation=45)
    ax.set_yticklabels([str(sampled_frames[i]) for i in tick_indices])
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Similarity matrix saved to: {output_file}")


def save_clusters_to_json(frame_numbers, clusters, cluster_info, output_file='clusters.json'):
    output_data = {
        'clustering_results': {
            'n_clusters': len([c for c in np.unique(clusters) if c != -1]),
            'n_noise': len(clusters[clusters == -1]),
            'total_frames': len(frame_numbers)
        },
        'frame_clusters': {str(fn): int(c) for fn, c in zip(frame_numbers, clusters)},
        'cluster_info': cluster_info
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nClustering results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Cluster CLIP embeddings using DBSCAN")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file (YAML)')
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Extract configuration parameters
    input_file = os.path.join(config["main"]["output_dir"], "embeddings", "frame_embeddings.json")
    eps = config["clustering"].get('eps', 0.3)
    min_samples = config["clustering"].get('min_samples', 3)
    output_json = os.path.join(config["main"]["output_dir"], "clustering", "clusters.json")
    output_viz = os.path.join(config["main"]["output_dir"], "clustering", "clusters_visualization.png")
    frame_numbers, embeddings = load_embeddings_from_json(input_file)
    
    clusters = perform_dbscan_clustering(embeddings, eps=eps, min_samples=min_samples)
    
    cluster_info = analyze_clusters(frame_numbers, clusters)
    
    # Save frames to cluster directories
    save_frames_by_cluster(frame_numbers, clusters, config)
    
    visualize_clusters(frame_numbers, clusters, output_viz)
    
    save_clusters_to_json(frame_numbers, clusters, cluster_info, output_json)


if __name__ == "__main__":
    main()