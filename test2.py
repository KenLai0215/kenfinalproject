import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def get_ranklists_by_id(df, query_id, k_runs=10):
    """Get ranklists for each run (seq_no from 1 to k_runs) for the specified query_id"""
    id_df = df[df['id'] == query_id]
    ranklists = []
    for seq in range(1, k_runs + 1):
        seq_df = id_df[id_df['seq_no'] == seq]
        ranklists.append(seq_df['page_id'].tolist())
    return ranklists

def rbo(list1, list2, p=0.9, k=None):
    """
    Compute the Rank-Biased Overlap (RBO) similarity score, see Webber 2010.
    list1, list2: ordered lists
    p: persistence parameter
    k: truncation depth
    Return value is in [0, 1].
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in the interval (0, 1)")
    S = list(list1)
    T = list(list2)
    if k is None:
        k = max(len(S), len(T))
    seen_S = set()
    seen_T = set()
    accum = 0.0
    for d in range(1, k + 1):
        if d <= len(S):
            seen_S.add(S[d - 1])
        if d <= len(T):
            seen_T.add(T[d - 1])
        overlap = len(seen_S & seen_T)
        accum += (overlap / d) * (p ** d)
    return (1 - p) * accum / p

def average_global_rbo(ranklists, p=0.9, k=None):
    """
    Compute the average global RBO score across ranklists for pages appearing in multiple lists.
    Combines the functionality of global_rbo_for_doc and average_global_rbo.
    """
    # Extract all unique page_ids from ranklists
    all_page_ids = []
    for ranklist in ranklists:
        all_page_ids.extend(ranklist)
    page_ids = list(set(all_page_ids))
    
    sum_rbo = 0
    count = 0
    
    for page_id in page_ids:
        # Find ranklists where this page_id appears
        valid_ranklists = [rnk for rnk in ranklists if page_id in rnk]
        
        if len(valid_ranklists) >= 2:  # Need at least 2 occurrences to compare
            scores = []
            for list1, list2 in combinations(valid_ranklists, 2):
                scores.append(rbo(list1, list2, p=p, k=k))
            
            if scores:  # If we have valid scores
                sum_rbo += sum(scores) / len(scores)
                count += 1
    
    return sum_rbo / count if count > 0 else None

def get_variance_statistics(df, query_id, k_runs=10):
    """
    Calculate variance statistics for page_ids that appear more than once.
    Optimized version that takes pre-filtered dataframe.
    """
    # Use the already filtered data
    sub_df = df[df['id'] == query_id]
    
    # Count occurrences of each page_id
    page_counts = sub_df['page_id'].value_counts()
    # Keep only page_ids that appear more than once
    valid_page_ids = page_counts[page_counts > 1].index.tolist()
    
    variances = []
    for page_id in valid_page_ids:
        ranks = []
        for seq_no in range(1, k_runs + 1):
            seq_df = sub_df[sub_df['seq_no'] == seq_no].reset_index(drop=True)
            if page_id in seq_df['page_id'].values:
                # Use the list position as the rank (starting from 1)
                rank = seq_df[seq_df['page_id'] == page_id].index[0] + 1
                ranks.append(int(rank))
        
        # Calculate the variance of ranks
        if len(ranks) > 1:
            var = np.var(ranks, ddof=1)
        else:
            var = 0
        variances.append(var)
    
    if len(variances) > 0:
        average_variance = float(np.mean(variances))
        min_variance = float(np.min(variances))
        max_variance = float(np.max(variances))
    else:
        average_variance = 0.0
        min_variance = 0.0
        max_variance = 0.0
    
    return average_variance, min_variance, max_variance

def analyze_single_configuration(df, unique_ids, p_value, k_value, k_runs):
    """
    Analyze a single parameter configuration and return aggregated results
    """
    rbo_scores = []
    avg_variances = []
    min_variances = []
    max_variances = []
    
    for query_id in unique_ids:
        # Get ranklists once and reuse
        ranklists = get_ranklists_by_id(df, query_id, k_runs=k_runs)
        
        # Calculate global RBO using simplified function
        global_rbo_score = average_global_rbo(ranklists, p=p_value, k=k_value)
        if global_rbo_score is not None:
            rbo_scores.append(global_rbo_score)
        
        # Calculate variance statistics
        average_variance, min_variance, max_variance = get_variance_statistics(df, query_id, k_runs=k_runs)
        avg_variances.append(average_variance)
        min_variances.append(min_variance)
        max_variances.append(max_variance)
    
    # Return aggregated statistics (removed std calculations)
    return {
        'mean_rbo': np.mean(rbo_scores) if rbo_scores else None,
        'mean_avg_variance': np.mean(avg_variances),
        'mean_min_variance': np.mean(min_variances),
        'mean_max_variance': np.mean(max_variances),
        'valid_queries': len(rbo_scores)
    }

def run_parameter_experiments():
    """
    Run comprehensive parameter experiments and save results
    """
    print("Starting parameter experiments...")
    
    # Read data
    df = pd.read_csv('stochastic_runs/input.UoGTrMabSAED', sep='\t')
    unique_ids = [int(i) for i in df['id'].unique().tolist()]
    
    # Create results directory
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    
    # Initialize results lists for different experiment types
    p_value_results = []
    k_value_results = []
    k_runs_results = []
    
    # Optimized parameter ranges for faster execution
    p_values = [0.3, 0.5, 0.7, 0.9]  
    k_values = [5, 10, 15, 20]  
    k_runs_values = [10, 25, 50, 75, 100]  
    
    total_experiments = len(p_values) + len(k_values) + len(k_runs_values)
    current_exp = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Total experiments to run: {total_experiments}")
    
    # Test p_value variations (k_value=None, k_runs=10)
    print("\n1. Testing p_value variations...")
    for p_val in p_values:
        current_exp += 1
        print(f"  Progress: {current_exp}/{total_experiments} - p_value={p_val}")
        
        result = analyze_single_configuration(df, unique_ids, p_val, None, 10)
        result.update({
            'experiment_type': 'p_value_test',
            'p_value': p_val,
            'k_value': None,
            'k_runs': 10
        })
        p_value_results.append(result)
    
    # Test k_value variations (p_value=0.9, k_runs=10)
    print("\n2. Testing k_value variations...")
    for k_val in k_values:
        current_exp += 1
        print(f"  Progress: {current_exp}/{total_experiments} - k_value={k_val}")
        
        result = analyze_single_configuration(df, unique_ids, 0.9, k_val, 10)
        result.update({
            'experiment_type': 'k_value_test',
            'p_value': 0.9,
            'k_value': k_val,
            'k_runs': 10
        })
        k_value_results.append(result)
    
    # Test k_runs variations (p_value=0.9, k_value=None)
    print("\n3. Testing k_runs variations...")
    for k_runs_val in k_runs_values:
        current_exp += 1
        print(f"  Progress: {current_exp}/{total_experiments} - k_runs={k_runs_val}")
        
        result = analyze_single_configuration(df, unique_ids, 0.9, None, k_runs_val)
        result.update({
            'experiment_type': 'k_runs_test',
            'p_value': 0.9,
            'k_value': None,
            'k_runs': k_runs_val
        })
        k_runs_results.append(result)
    
    # Save results to separate CSV files
    csv_files = {}
    
    # Save p_value test results
    p_value_df = pd.DataFrame(p_value_results)
    p_value_csv = f'experiment_results/p_value_test_{timestamp}.csv'
    p_value_df.to_csv(p_value_csv, index=False)
    csv_files['p_value'] = p_value_csv
    print(f"P-value results saved to: {p_value_csv}")
    
    # Save k_value test results
    k_value_df = pd.DataFrame(k_value_results)
    k_value_csv = f'experiment_results/k_value_test_{timestamp}.csv'
    k_value_df.to_csv(k_value_csv, index=False)
    csv_files['k_value'] = k_value_csv
    print(f"K-value results saved to: {k_value_csv}")
    
    # Save k_runs test results
    k_runs_df = pd.DataFrame(k_runs_results)
    k_runs_csv = f'experiment_results/k_runs_test_{timestamp}.csv'
    k_runs_df.to_csv(k_runs_csv, index=False)
    csv_files['k_runs'] = k_runs_csv
    print(f"K-runs results saved to: {k_runs_csv}")
    
    # Combine all results for visualization
    all_results = p_value_results + k_value_results + k_runs_results
    results_df = pd.DataFrame(all_results)
    
    return results_df, csv_files, timestamp

def create_visualizations(results_df, csv_files, timestamp):
    """
    Create comprehensive visualizations for the parameter experiments
    """
    print("\nCreating visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure directory
    fig_dir = 'experiment_results/figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # 1. P-value effects (only RBO, variance is not affected by p_value)
    print("  Creating p-value effect plots...")
    p_value_data = results_df[results_df['experiment_type'] == 'p_value_test']
    
    plt.figure(figsize=(10, 6))
    
    # RBO vs p_value
    plt.plot(p_value_data['p_value'], p_value_data['mean_rbo'], 'bo-', linewidth=3, markersize=8)
    plt.xlabel('P-value (Persistence Parameter)', fontsize=12)
    plt.ylabel('Mean RBO Score', fontsize=12)
    plt.title('RBO Score vs P-value', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/p_value_effects_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. K-value effects (only RBO, variance is not affected by k_value)
    print("  Creating k-value effect plots...")
    k_value_data = results_df[results_df['experiment_type'] == 'k_value_test']
    
    plt.figure(figsize=(10, 6))
    
    # RBO vs k_value
    plt.plot(k_value_data['k_value'], k_value_data['mean_rbo'], 'go-', linewidth=3, markersize=8)
    plt.xlabel('K-value (Truncation Depth)', fontsize=12)
    plt.ylabel('Mean RBO Score', fontsize=12)
    plt.title('RBO Score vs K-value', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/k_value_effects_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. K-runs effects (RBO and variance both affected)
    print("  Creating k-runs effect plots...")
    k_runs_data = results_df[results_df['experiment_type'] == 'k_runs_test']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RBO vs k_runs
    ax1.plot(k_runs_data['k_runs'], k_runs_data['mean_rbo'], 'co-', linewidth=3, markersize=8)
    ax1.set_xlabel('K-runs (Number of Runs)', fontsize=12)
    ax1.set_ylabel('Mean RBO Score', fontsize=12)
    ax1.set_title('RBO Score vs K-runs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Variance vs k_runs
    ax2.plot(k_runs_data['k_runs'], k_runs_data['mean_avg_variance'], 'mo-', linewidth=3, markersize=8)
    ax2.set_xlabel('K-runs (Number of Runs)', fontsize=12)
    ax2.set_ylabel('Mean Average Variance', fontsize=12)
    ax2.set_title('Average Variance vs K-runs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/k_runs_effects_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Summary heatmap
    print("  Creating summary heatmap...")
    
    # Create correlation matrix for key metrics
    summary_data = results_df[['p_value', 'k_value', 'k_runs', 'mean_rbo', 'mean_avg_variance']].copy()
    summary_data['k_value'] = summary_data['k_value'].fillna(0)  # Replace None with 0 for correlation
    
    correlation_matrix = summary_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/correlation_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"All visualizations saved to: {fig_dir}/")
    
    return fig_dir

def main():
    """
    Main function - run experiments and create visualizations
    """
    print("=== RBO Parameter Analysis Experiment ===")
    
    # Run comprehensive experiments
    results_df, csv_files, timestamp = run_parameter_experiments()
    
    # Create visualizations
    fig_dir = create_visualizations(results_df, csv_files, timestamp)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results CSV files:")
    print(f"  - P-value test: {csv_files['p_value']}")
    print(f"  - K-value test: {csv_files['k_value']}")
    print(f"  - K-runs test: {csv_files['k_runs']}")
    print(f"Figures: {fig_dir}/")
    print(f"Total experiments conducted: {len(results_df)}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main()