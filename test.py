import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

# Set English font (remove Chinese font settings)
matplotlib.rcParams['axes.unicode_minus'] = False  # Correctly display minus sign


def analyze_page_position_variance(file_path, k_threshold=2):
    """
    Analyze the variance of page_id positions across different rounds for each query.

    Args:
    file_path: Path to the data file
    k_threshold: Minimum number of occurrences for a page_id within the same query

    Returns:
    A dictionary containing variance information for each qualifying page_id and the average variance for each query
    """
    print(f"Reading data file: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Data shape: {df.shape}")
    print(f"First 5 rows:")
    print(df.head())

    results = {}
    query_average_variances = {}  # Store average variance for each query

    # Process by query (formerly user)
    for query_id in df['id'].unique():
        print(f"\nProcessing Query ID: {query_id}")
        query_data = df[df['id'] == query_id]

        # Count occurrences of each page_id under this query
        page_counts = query_data['page_id'].value_counts()

        # Find page_ids that appear at least k_threshold times
        frequent_pages = page_counts[page_counts >= k_threshold].index.tolist()
        print(f"Number of page_ids appearing at least {k_threshold} times: {len(frequent_pages)}")

        if len(frequent_pages) == 0:
            print(f"Query {query_id} has no qualifying page_id")
            query_average_variances[query_id] = 0  # No qualifying page_id, average variance is 0
            continue

        # Calculate position variance for each qualifying page_id
        page_variances = {}
        variances_for_average = []  # For calculating average variance for this query

        for page_id in frequent_pages:
            positions = []

            # Get the position of this page_id in each seq_no
            for seq_no in sorted(query_data['seq_no'].unique()):
                seq_data = query_data[query_data['seq_no'] == seq_no]
                seq_pages = seq_data['page_id'].tolist()

                if page_id in seq_pages:
                    # Position starts from 1
                    position = seq_pages.index(page_id) + 1
                    positions.append(position)

            # Calculate variance
            if len(positions) >= 2:  # At least 2 positions needed to calculate variance
                variance = np.var(positions, ddof=1)  # Sample variance
                mean_position = np.mean(positions)

                page_variances[page_id] = {
                    'positions': positions,
                    'mean_position': mean_position,
                    'variance': variance,
                    'std_dev': np.sqrt(variance),
                    'occurrence_count': len(positions)
                }

                variances_for_average.append(variance)
                # print(f"  Page ID {page_id}: positions {positions}, mean position {mean_position:.2f}, variance {variance:.2f}")

        # Calculate average variance for this query
        if variances_for_average:
            avg_variance = np.mean(variances_for_average)
            query_average_variances[query_id] = avg_variance
            print(f"Query {query_id} average variance: {avg_variance:.4f}")
        else:
            query_average_variances[query_id] = 0
            print(f"Query {query_id} has no page_id with calculable variance")

        results[query_id] = page_variances

    return results, query_average_variances


def plot_query_variances(query_average_variances, k_threshold):
    """
    Plot the average variance for each query.

    Args:
    query_average_variances: Dictionary of average variance for each query
    k_threshold: K threshold value
    """
    # Filter out valid variance values (>0)
    valid_data = [(query_id, var) for query_id, var in query_average_variances.items() if var > 0]

    if not valid_data:
        print("No valid variance data available for plotting.")
        return

    # Sort by query ID
    valid_data.sort(key=lambda x: x[0])

    query_ids = [item[0] for item in valid_data]
    variances = [item[1] for item in valid_data]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Use scatter plot if there are too many queries; otherwise use bar chart
    if len(query_ids) > 20:
        plt.scatter(query_ids, variances, alpha=0.6, s=50)
        plt.plot(query_ids, variances, alpha=0.3, linewidth=1)
        chart_type = "Scatter Plot"
    else:
        bars = plt.bar(range(len(query_ids)), variances, alpha=0.7)
        plt.xticks(range(len(query_ids)), query_ids, rotation=45)
        chart_type = "Bar Chart"

        # Add value labels on the bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Query ID', fontsize=12)
    plt.ylabel('Average Variance', fontsize=12)
    plt.title(f'Average Variance Distribution per Query ({chart_type}, K threshold={k_threshold})', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add statistical lines to the plot
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    plt.axhline(y=mean_var, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_var:.3f}')
    plt.axhline(y=mean_var + std_var, color='orange', linestyle=':', alpha=0.7,
                label=f'Mean+Std: {mean_var + std_var:.3f}')
    plt.axhline(y=mean_var - std_var, color='orange', linestyle=':', alpha=0.7,
                label=f'Mean-Std: {mean_var - std_var:.3f}')

    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig('query_average_variances.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as 'query_average_variances.png'")

    # Show the plot
    plt.show()

    # Print plot statistics
    print(f"\nPlot statistics:")
    print(f"  Number of valid queries: {len(query_ids)}")
    print(f"  Range of average variance: {min(variances):.4f} - {max(variances):.4f}")
    print(f"  Mean of average variance: {mean_var:.4f}")
    print(f"  Std of average variance: {std_var:.4f}")


def display_summary(results, query_average_variances, k_threshold):
    """Display summary of analysis results"""
    print(f"\n{'=' * 60}")
    print(f"Analysis Summary (K threshold = {k_threshold})")
    print(f"{'=' * 60}")

    total_queries = len(results)
    total_pages = sum(len(pages) for pages in results.values())

    print(f"Number of queries analyzed: {total_queries}")
    print(f"Total number of qualifying page_ids: {total_pages}")

    # Show average variance for each query
    print(f"\nAverage variance for each query:")
    print(f"{'Query ID':<15} {'Average Variance':<20} {'Number of Qualifying Pages':<30}")
    print("-" * 65)

    valid_avg_variances = []  # Only collect valid average variances (>0)

    for query_id in sorted(query_average_variances.keys()):
        avg_var = query_average_variances[query_id]
        page_count = len(results.get(query_id, {}))
        print(f"{query_id:<15} {avg_var:<20.4f} {page_count:<30}")

        if avg_var > 0:
            valid_avg_variances.append(avg_var)

    # Show statistics for query average variances
    if valid_avg_variances:
        print(f"\nQuery average variance statistics:")
        print(f"  Number of valid queries (average variance > 0): {len(valid_avg_variances)}")
        print(f"  Minimum average variance: {min(valid_avg_variances):.4f}")
        print(f"  Maximum average variance: {max(valid_avg_variances):.4f}")
        print(f"  Mean of all query average variances: {np.mean(valid_avg_variances):.4f}")
        print(f"  Std of all query average variances: {np.std(valid_avg_variances):.4f}")

        # Show top 5 queries with highest average variance
        sorted_queries = sorted([(query_id, var) for query_id, var in query_average_variances.items() if var > 0],
                                key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 queries with highest average variance:")
        for i, (query_id, avg_var) in enumerate(sorted_queries[:5]):
            page_count = len(results.get(query_id, {}))
            print(
                f"  {i + 1}. Query ID {query_id}: Average Variance={avg_var:.4f}, Number of Qualifying Pages={page_count}")

    if total_pages > 0:
        # Collect all variance values
        all_variances = []
        for query_id, pages in results.items():
            for page_id, stats in pages.items():
                all_variances.append(stats['variance'])

        print(f"\nStatistics for all page variances:")
        print(f"  Minimum variance: {min(all_variances):.4f}")
        print(f"  Maximum variance: {max(all_variances):.4f}")
        print(f"  Mean variance: {np.mean(all_variances):.4f}")
        print(f"  Std of variances: {np.std(all_variances):.4f}")

        # Show top 5 page_ids with highest variance
        variance_list = []
        for query_id, pages in results.items():
            for page_id, stats in pages.items():
                variance_list.append((query_id, page_id, stats['variance'], stats['positions']))

        variance_list.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop 5 page_ids with highest variance:")
        for i, (query_id, page_id, variance, positions) in enumerate(variance_list[:5]):
            print(f"  {i + 1}. Query {query_id}, Page {page_id}: Variance={variance:.4f}, Positions={positions}")


# Main program
if __name__ == "__main__":
    file_path = 'stochastic_runs\\input.UoGTrMabSAED'

    # You can adjust K value
    K = 2  # Default is 2, modify as needed

    print(f"Start analysis, K threshold set to: {K}")

    # Run analysis
    results, query_average_variances = analyze_page_position_variance(file_path, k_threshold=K)

    # Display summary
    display_summary(results, query_average_variances, K)

    # Plot chart
    print(f"\nGenerating chart...")
    plot_query_variances(query_average_variances, K)

