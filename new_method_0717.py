import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import random
from matplotlib.patches import Rectangle

# Set font for English and Unicode support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NewMethod:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_path, self.timestamp)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.results_dir = os.path.join(self.output_dir, "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Read data and set correct column names
        self.data = pd.read_csv(self.data_path, sep='\t')
        self.seq_no_list = [int(x) for x in self.data['seq_no'].unique()]
        self.id_list = [int(x) for x in self.data['id'].unique()]
        self.seq_dict = self.split_rank_lists()
        self.top_k = None
        self.split_size = 0.8
        self.k_fold = 10
        self.random_seed = 42
        self.k_runs = 100
        self.min_samples = 3
        self.min_variance = 0.01
        self.log_likelihood_floor = -5.0
        self.use_laplace_smoothing = True
        random.seed(self.random_seed)
        
        # For storing visualization data
        self.visualization_data = {}
        
    def split_rank_lists(self):
        """
        Split the ranking lists by seq_no and store them in a dictionary.
        Returns a dictionary: key is seq_no, value is a DataFrame containing query_id, seq_no, doc_id.
        """
        seq_no_list = [int(x) for x in self.data['seq_no'].unique()]
        self.seq_dict = {}
        for seq_no in seq_no_list:
            df_seq = self.data[self.data['seq_no'] == seq_no]
            self.seq_dict[seq_no] = df_seq
        return self.seq_dict
    
    def split_train_test_k_fold(self):
        """
        Randomly split the data into k folds, returning k groups, each containing train and test seq_no lists.
        Ensures that each fold split is unique.
        """
        if not (0 < self.split_size <= 1):
            raise ValueError("split_size must be in (0, 1].")
        seq_no_list = self.seq_no_list[:self.k_runs]
        fold_results = {}
        seen = set()
        for fold in range(self.k_fold):
            for _ in range(1000):  # Try up to 1000 times
                shuffled_seq_no = seq_no_list[:]
                random.shuffle(shuffled_seq_no)
                split_idx = int(len(shuffled_seq_no) * self.split_size)
                train_seq_no_list = shuffled_seq_no[:split_idx]
                test_seq_no_list = shuffled_seq_no[split_idx:]
                fold_signature = (tuple(sorted(train_seq_no_list)), tuple(sorted(test_seq_no_list)))
                if fold_signature not in seen:
                    seen.add(fold_signature)
                    fold_results[fold] = {
                        "train": list(train_seq_no_list),
                        "test": list(test_seq_no_list)
                    }
                    break
            else:
                raise RuntimeError("Unable to generate a new unique fold split.")
        return fold_results

    def get_page_id_from_id(self, id):
        """
        For a given id, return all page_ids and their ranks (starting from 1) under different seq_no.
        Returns a dictionary: key is seq_no, value is a dict {page_id: rank}.
        Only keeps page_ids within top_k. If self.top_k is None, use all.
        """
        result = {}
        for seq_no, df in self.seq_dict.items():
            df_id = df[df['id'] == id]
            if self.top_k is None:
                df_id_topk = df_id
            else:
                df_id_topk = df_id.head(self.top_k)
            page_id_rank = {page_id: rank for rank, page_id in enumerate(df_id_topk['page_id'], 1)}
            if page_id_rank:  # Only add if there are results
                result[seq_no] = page_id_rank
        return result

    def merge_page_id_dict(self, id, fold, fold_results):
        """
        Merge multiple dictionaries, return a dictionary: key is page_id, value is a list of ranks.
        """
        trainresult = {}
        testresult = {}
        page_id_dict = self.get_page_id_from_id(id)
        for seq_no in fold_results[fold]['train']:
            if seq_no not in page_id_dict:
                continue
            for page_id, rank in page_id_dict[seq_no].items():
                if page_id not in trainresult:
                    trainresult[page_id] = []
                trainresult[page_id].append(rank)
        for seq_no in fold_results[fold]['test']:
            if seq_no not in page_id_dict:
                continue
            for page_id, rank in page_id_dict[seq_no].items():
                if page_id not in testresult:
                    testresult[page_id] = []
                testresult[page_id].append(rank)
        return trainresult, testresult

    def gaussian_score_from_trainresult(self, trainresult):
        """
        Calculate Gaussian score from trainresult, with Laplace smoothing.
        """
        doc_stats = {}
        for page_id, ranks in trainresult.items():
            if len(ranks) < self.min_samples:
                continue
            ranks = np.array(ranks)
            mu = np.mean(ranks)
            if self.use_laplace_smoothing:
                # Laplace smoothing: add a minimum value to variance to prevent zero variance
                sigma2 = np.var(ranks) + self.min_variance
            else:
                sigma2 = np.var(ranks)
            doc_stats[page_id] = {'mu': mu, 'sigma2': sigma2}
        return doc_stats

    def cal_for_single_page_id_log_likelihood(self, test_ranks, mu, sigma2):
        # Prevent division by zero if variance is zero
        sigma2 = max(sigma2, self.min_variance)
        log_likes = []
        for x_test in test_ranks:
            ll = -0.5 * np.log(2 * np.pi * sigma2) - ((x_test - mu) ** 2) / (2 * sigma2)
            log_likes.append(ll)
        return log_likes  # Return as list

    def predict_single_id_qpp_from_testresult(self, testresult, doc_stats):
        all_log_likes = []
        matched_page_id = False  # Flag to indicate if any page_id matched
        
        for page_id, test_ranks in testresult.items():
            if page_id in doc_stats:
                matched_page_id = True
                mu = doc_stats[page_id]['mu']
                sigma2 = doc_stats[page_id]['sigma2']
                log_likes = self.cal_for_single_page_id_log_likelihood(test_ranks, mu, sigma2)
                all_log_likes.extend(log_likes)
        
        if not all_log_likes:
            # If no matching page_id found, return the floor score
            qpp_score = self.log_likelihood_floor
        else:
            qpp_score = np.mean(all_log_likes)
        
        # Final numerical stability check
        if np.isnan(qpp_score) or np.isinf(qpp_score):
            qpp_score = self.log_likelihood_floor
            
        return qpp_score

    def all_id_qpp_score(self):
        merged_df = pd.DataFrame()
        self.fold_results = self.split_train_test_k_fold()
        all_fold_dfs = []  # For collecting DataFrames from each fold
        for fold in range(self.k_fold):
            results = []
            print(f"Running QPP score calculation for fold {fold + 1}, start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            start_time = time.time()
            for id in self.id_list:
                trainresult, testresult = self.merge_page_id_dict(id, fold, self.fold_results)
                doc_stats = self.gaussian_score_from_trainresult(trainresult)
                qpp_score = self.predict_single_id_qpp_from_testresult(testresult, doc_stats)
                results.append({
                    "top_k": self.top_k,
                    "split_size": self.split_size,
                    "k_fold": self.k_fold,
                    "k_runs": self.k_runs,
                    "fold": fold,
                    "id": id,
                    "trainresult": trainresult,
                    "testresult": testresult,
                    "qpp_score": qpp_score,
                    "min_samples": self.min_samples,
                    "min_variance": self.min_variance,
                    "log_likelihood_floor": self.log_likelihood_floor,
                    "use_laplace_smoothing": self.use_laplace_smoothing
                })
            df = pd.DataFrame(results)
            output_csv_path = os.path.join(self.results_dir, f"qpp_scores_fold_{fold}_{self.timestamp}.csv")
            df.to_csv(output_csv_path, index=False, encoding="utf-8")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Fold {fold + 1} QPP score calculation finished, elapsed time: {elapsed_time:.2f} seconds")
            print(f"Results saved to {output_csv_path}")
            print("--------------------------------")
            if not df.empty:
                all_fold_dfs.append(df)
        if all_fold_dfs:
            merged_df = pd.concat(all_fold_dfs, ignore_index=True)
            merged_df.to_csv(os.path.join(self.results_dir, f"qpp_scores_all_{self.timestamp}.csv"), index=False, encoding="utf-8")
        else:
            print("No valid fold results, cannot merge.")
        # Calculate mean, max, min QPP score across 10 folds
        if not merged_df.empty and "qpp_score" in merged_df.columns:
            avg_qpp_score = merged_df["qpp_score"].mean()
            max_qpp_score = merged_df["qpp_score"].max()
            min_qpp_score = merged_df["qpp_score"].min()
            print(f"Mean QPP score across 10 folds: {avg_qpp_score}")
            print(f"Max QPP score across 10 folds: {max_qpp_score}")
            print(f"Min QPP score across 10 folds: {min_qpp_score}")
        else:
            print("No QPP score data, cannot calculate mean, max, and min.")
        
        # Automatically generate visualization report
        if not merged_df.empty:
            print("\nGenerating visualization analysis...")
            self.create_comprehensive_report(merged_df)
        
        return merged_df
            
    def plot_qpp_score_distribution(self, merged_df, save_path=None):
        """
        Plot QPP score distribution (all plots in English)
        """
        if merged_df.empty or "qpp_score" not in merged_df.columns:
            print("No QPP score data available, cannot plot distribution.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'QPP Score Analysis (Timestamp: {self.timestamp})', fontsize=16, fontweight='bold')

        # 1. Overall QPP score histogram
        axes[0, 0].hist(merged_df['qpp_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        mean_qpp = merged_df['qpp_score'].mean()
        axes[0, 0].axvline(mean_qpp, color='red', linestyle='--',
                           label='Mean: {:.3f}'.format(mean_qpp))
        axes[0, 0].set_title('Overall QPP Score Distribution')
        axes[0, 0].set_xlabel('QPP Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. QPP score boxplot for each fold
        fold_scores = [merged_df[merged_df['fold'] == fold]['qpp_score'].values
                       for fold in range(self.k_fold)]
        bp = axes[0, 1].boxplot(fold_scores, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0, 1].set_title('QPP Score Distribution by Fold')
        axes[0, 1].set_xlabel('Fold Index')
        axes[0, 1].set_ylabel('QPP Score')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Top queries by average QPP score
        top_queries = merged_df.groupby('id')['qpp_score'].mean().nlargest(10)
        axes[1, 0].bar(range(len(top_queries)), top_queries.values, color='lightgreen')
        axes[1, 0].set_title('Top 10 Queries by Mean QPP Score')
        axes[1, 0].set_xlabel('Query Rank')
        axes[1, 0].set_ylabel('Mean QPP Score')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Fold stability analysis
        fold_means = merged_df.groupby('fold')['qpp_score'].mean()
        axes[1, 1].plot(fold_means.index, fold_means.values, marker='o', linewidth=2, markersize=8)
        axes[1, 1].axhline(fold_means.mean(), color='red', linestyle='--',
                           label=f'Overall Mean: {fold_means.mean():.3f}')
        axes[1, 1].set_title('QPP Score Stability Across Folds')
        axes[1, 1].set_xlabel('Fold Index')
        axes[1, 1].set_ylabel('Mean QPP Score per Fold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, f'qpp_score_analysis_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'QPP score analysis plot saved to: {save_path}')

    def plot_parameter_sensitivity(self, merged_df, save_path=None):
        """
        Plot parameter sensitivity analysis (all plots in English)
        """
        if merged_df.empty:
            print("No data available, cannot perform parameter sensitivity analysis.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Parameter Sensitivity Analysis (Timestamp: {self.timestamp})', fontsize=16, fontweight='bold')

        # 1. Variance analysis for different queries
        query_stats = merged_df.groupby('id')['qpp_score'].agg(['mean', 'std', 'count'])
        query_stats = query_stats[query_stats['count'] >= 5]  # Only queries with enough data

        scatter = axes[0, 0].scatter(query_stats['mean'], query_stats['std'],
                                     alpha=0.6, s=60, c=query_stats['count'], cmap='viridis')
        axes[0, 0].set_title('QPP Score Mean vs. Std for Query IDs')
        axes[0, 0].set_xlabel('Mean QPP Score')
        axes[0, 0].set_ylabel('Std of QPP Score')
        plt.colorbar(scatter, ax=axes[0, 0], label='Number of Data Points')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. QPP score trend across folds
        fold_stats = merged_df.groupby('fold')['qpp_score'].agg(['mean', 'std'])
        axes[0, 1].errorbar(fold_stats.index, fold_stats['mean'], yerr=fold_stats['std'],
                            marker='o', linewidth=2, markersize=8, capsize=5)
        axes[0, 1].set_title('QPP Score Trend Across Folds')
        axes[0, 1].set_xlabel('Fold Index')
        axes[0, 1].set_ylabel('QPP Score (Mean ± Std)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Outlier detection (IQR-based)
        Q1 = merged_df['qpp_score'].quantile(0.25)
        Q3 = merged_df['qpp_score'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = merged_df[(merged_df['qpp_score'] < Q1 - 1.5 * IQR) |
                             (merged_df['qpp_score'] > Q3 + 1.5 * IQR)]

        if not outliers.empty:
            axes[1, 0].scatter(outliers['id'], outliers['qpp_score'],
                               color='red', alpha=0.7, s=60, label=f'Outliers ({len(outliers)})')
            normal_data = merged_df[~merged_df.index.isin(outliers.index)]
            axes[1, 0].scatter(normal_data['id'], normal_data['qpp_score'],
                               color='blue', alpha=0.3, s=20, label='Normal')
        else:
            axes[1, 0].scatter(merged_df['id'], merged_df['qpp_score'],
                               color='blue', alpha=0.5, s=30)

        axes[1, 0].set_title('Outlier Detection (IQR-based)')
        axes[1, 0].set_xlabel('Query ID')
        axes[1, 0].set_ylabel('QPP Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Parameter settings and data statistics
        param_text = f"""Current Parameter Settings:
        • top_k: {self.top_k}
        • split_size: {self.split_size}
        • k_fold: {self.k_fold}
        • k_runs: {self.k_runs}
        • min_samples: {self.min_samples}
        • min_variance: {self.min_variance}
        • log_likelihood_floor: {self.log_likelihood_floor}
        • use_laplace_smoothing: {self.use_laplace_smoothing}

        Data Statistics:
        • Number of queries: {len(self.id_list)}
        • Number of sequences: {len(self.seq_no_list)}
        • QPP score range: [{merged_df['qpp_score'].min():.3f}, {merged_df['qpp_score'].max():.3f}]
        • QPP score mean: {merged_df['qpp_score'].mean():.3f}
        • QPP score std: {merged_df['qpp_score'].std():.3f}
        """

        axes[1, 1].text(0.05, 0.95, param_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Experiment Parameters & Data Statistics')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, f'parameter_sensitivity_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Parameter sensitivity analysis plot saved to: {save_path}')

    def create_comprehensive_report(self, merged_df):
        """
        Create a comprehensive analysis report, including all visualizations (plots in English)
        """
        print(f"\n{'='*60}")
        print(f"Generating comprehensive analysis report...")
        print(f"{'='*60}")

        # Generate all visualizations
        self.plot_qpp_score_distribution(merged_df)
        self.plot_parameter_sensitivity(merged_df)

        # Generate textual report (in English)
        report_path = os.path.join(self.output_dir, f'analysis_report_{self.timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"QPP Method Analysis Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"Data path: {self.data_path}\n")
            f.write(f"Output path: {self.output_dir}\n\n")

            f.write(f"Experiment Parameters:\n")
            f.write(f"- top_k: {self.top_k}\n")
            f.write(f"- split_size: {self.split_size}\n")
            f.write(f"- k_fold: {self.k_fold}\n")
            f.write(f"- k_runs: {self.k_runs}\n")
            f.write(f"- min_samples: {self.min_samples}\n")
            f.write(f"- min_variance: {self.min_variance}\n")
            f.write(f"- log_likelihood_floor: {self.log_likelihood_floor}\n")
            f.write(f"- use_laplace_smoothing: {self.use_laplace_smoothing}\n\n")

            if not merged_df.empty:
                f.write(f"QPP Score Statistics:\n")
                f.write(f"- Total samples: {len(merged_df)}\n")
                f.write(f"- Mean: {merged_df['qpp_score'].mean():.6f}\n")
                f.write(f"- Std: {merged_df['qpp_score'].std():.6f}\n")
                f.write(f"- Min: {merged_df['qpp_score'].min():.6f}\n")
                f.write(f"- Max: {merged_df['qpp_score'].max():.6f}\n")
                f.write(f"- Median: {merged_df['qpp_score'].median():.6f}\n\n")

                # Fold statistics
                fold_stats = merged_df.groupby('fold')['qpp_score'].agg(['mean', 'std'])
                f.write(f"Fold Statistics:\n")
                for fold in range(self.k_fold):
                    if fold in fold_stats.index:
                        f.write(f"- Fold {fold}: mean={fold_stats.loc[fold, 'mean']:.6f}, "
                                f"std={fold_stats.loc[fold, 'std']:.6f}\n")

        print(f"Comprehensive analysis report saved to: {report_path}")
        print(f"All visualizations saved to: {self.output_dir}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    data_path = os.path.join("stochastic_runs", "input.UoGTrMabSAED")
    output_path = "output"
    
    # Create an instance of the new method
    new_method = NewMethod(data_path, output_path)
    
    # Run the full QPP analysis (including automatic visualization)
    merged_df = new_method.all_id_qpp_score()
    
    
    # Debug code example (commented out)
    """
    test_id = 1018
    fold = 0
    fold_results = new_method.split_train_test_k_fold()
    print("fold_results", fold_results)
    
    trainresult, testresult = new_method.merge_page_id_dict(test_id, fold, fold_results)
    doc_stats = new_method.gaussian_score_from_trainresult(trainresult)
    qpp_score = new_method.predict_single_id_qpp_from_testresult(testresult, doc_stats)
    print("trainresult", trainresult)
    print("testresult", testresult)
    print("doc_stats", doc_stats)
    print("qpp_score", qpp_score)
    """
