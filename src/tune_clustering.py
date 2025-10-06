#!/usr/bin/env python3

import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from collections import Counter


def load_embeddings_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    frame_numbers = sorted([int(k) for k in data.keys()])
    embeddings = np.array([data[str(fn)] for fn in frame_numbers])
    
    return frame_numbers, embeddings


def test_dbscan_parameters(embeddings, eps_range, min_samples_range, target_clusters=2):
    normalized_embeddings = normalize(embeddings, norm='l2')
    results = []
    
    print(f"Testing DBSCAN parameters to find {target_clusters} clusters...")
    print("eps\tmin_samples\tn_clusters\tn_noise\tlargest_cluster_size")
    print("-" * 60)
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            clusters = dbscan.fit_predict(normalized_embeddings)
            
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters[unique_clusters != -1])
            n_noise = len(clusters[clusters == -1])
            
            cluster_counts = Counter(clusters[clusters != -1])
            largest_cluster_size = max(cluster_counts.values()) if cluster_counts else 0
            
            result = {
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'largest_cluster_size': largest_cluster_size,
                'clusters': clusters,
                'cluster_counts': dict(cluster_counts)
            }
            results.append(result)
            
            print(f"{eps:.3f}\t{min_samples}\t\t{n_clusters}\t\t{n_noise}\t\t{largest_cluster_size}")
    
    return results


def find_best_parameters(results, target_clusters=2):
    target_results = [r for r in results if r['n_clusters'] == target_clusters]
    
    if not target_results:
        print(f"\nNo parameter combination found with exactly {target_clusters} clusters")
        print("Available cluster counts:")
        cluster_counts = {}
        for r in results:
            count = r['n_clusters']
            if count not in cluster_counts:
                cluster_counts[count] = []
            cluster_counts[count].append(r)
        
        for count in sorted(cluster_counts.keys()):
            print(f"  {count} clusters: {len(cluster_counts[count])} parameter combinations")
        
        return None
    
    print(f"\nFound {len(target_results)} parameter combinations with {target_clusters} clusters:")
    
    best_result = None
    best_score = float('inf')
    
    for result in target_results:
        noise_penalty = result['n_noise']
        balance_penalty = abs(result['largest_cluster_size'] - len(result['clusters']) // target_clusters)
        score = noise_penalty + balance_penalty * 0.5
        
        print(f"  eps={result['eps']:.3f}, min_samples={result['min_samples']}, "
              f"noise={result['n_noise']}, largest_cluster={result['largest_cluster_size']}, "
              f"score={score:.1f}")
        
        if score < best_score:
            best_score = score
            best_result = result
    
    return best_result


def analyze_two_clusters(frame_numbers, clusters):
    cluster_0_frames = [frame_numbers[i] for i in range(len(frame_numbers)) if clusters[i] == 0]
    cluster_1_frames = [frame_numbers[i] for i in range(len(frame_numbers)) if clusters[i] == 1]
    noise_frames = [frame_numbers[i] for i in range(len(frame_numbers)) if clusters[i] == -1]
    
    print(f"\nTwo-Cluster Analysis:")
    print(f"Cluster 0: {len(cluster_0_frames)} frames")
    print(f"  Range: {min(cluster_0_frames) if cluster_0_frames else 'N/A'} - {max(cluster_0_frames) if cluster_0_frames else 'N/A'}")
    print(f"  Frames: {sorted(cluster_0_frames)[:10]}{'...' if len(cluster_0_frames) > 10 else ''}")
    
    print(f"Cluster 1: {len(cluster_1_frames)} frames")
    print(f"  Range: {min(cluster_1_frames) if cluster_1_frames else 'N/A'} - {max(cluster_1_frames) if cluster_1_frames else 'N/A'}")
    print(f"  Frames: {sorted(cluster_1_frames)[:10]}{'...' if len(cluster_1_frames) > 10 else ''}")
    
    print(f"Noise: {len(noise_frames)} frames")
    print(f"  Frames: {sorted(noise_frames)}")
    
    return {
        'cluster_0': sorted(cluster_0_frames),
        'cluster_1': sorted(cluster_1_frames),
        'noise': sorted(noise_frames)
    }


def main():
    parser = argparse.ArgumentParser(description='Tune DBSCAN parameters for optimal clustering')
    parser.add_argument('--input', type=str, default='frame_embeddings.json',
                       help='Input JSON file with frame embeddings')
    parser.add_argument('--target-clusters', type=int, default=2,
                       help='Target number of clusters')
    parser.add_argument('--eps-min', type=float, default=0.02,
                       help='Minimum eps value to test')
    parser.add_argument('--eps-max', type=float, default=0.12,
                       help='Maximum eps value to test')
    parser.add_argument('--eps-steps', type=int, default=20,
                       help='Number of eps values to test')
    parser.add_argument('--min-samples', type=int, nargs='+', default=[2, 3, 4, 5],
                       help='Min samples values to test')
    
    args = parser.parse_args()
    
    frame_numbers, embeddings = load_embeddings_from_json(args.input)
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    eps_range = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
    
    results = test_dbscan_parameters(
        embeddings, 
        eps_range, 
        args.min_samples, 
        target_clusters=args.target_clusters
    )
    
    best_result = find_best_parameters(results, target_clusters=args.target_clusters)
    
    if best_result:
        print(f"\nBest parameters:")
        print(f"  eps: {best_result['eps']:.3f}")
        print(f"  min_samples: {best_result['min_samples']}")
        print(f"  n_clusters: {best_result['n_clusters']}")
        print(f"  n_noise: {best_result['n_noise']}")
        
        if args.target_clusters == 2:
            cluster_analysis = analyze_two_clusters(frame_numbers, best_result['clusters'])
            
            print(f"\nRecommended command:")
            print(f"python cluster_embeddings.py --input {args.input} --eps {best_result['eps']:.3f} --min-samples {best_result['min_samples']}")


if __name__ == "__main__":
    main()