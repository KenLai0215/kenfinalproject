import pandas as pd
import data_loader
import constants
import pyterrier as pt
from trec_fair.tf_collection import OptimizedDocumentRetriever
import os
from qpp_methods.nqc_specificity import NQCSpecificity
from qpp_methods.uef_specificity import UEFSpecificity
from qpp_methods.rsd_specificity import RSDSpecificity
from evaluator import Evaluator
import numpy as np
from scipy.stats import kendalltau
import time
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# Compute missing scores and save to CSV file first
def compute_missing_scores(index, ranking_lists_df, queries_df, res_df, file_name=None):
    """
    Compute and fill missing document scores
    
    Args:
        index: PyTerrier index
        ranking_lists_df: Ranking list data (qid, seq_no, docno)
        queries_df: Query data (qid, query)
        res_df: Retrieval result data (qid, docno, score, ...)
    
    Returns:
        pd.DataFrame: Complete data (qid, seq_no, docno, rank, score)
    """
    
    # Add rank column to ranking_lists_df
    ranking_lists_with_rank = ranking_lists_df.copy()
    ranking_lists_with_rank['rank'] = ranking_lists_with_rank.groupby(['seq_no', 'qid']).cumcount() + 1
    
    # Find existing scores from res_df
    res_df_scores = res_df[['qid', 'docno', 'score']].copy()
    
    merged_df = ranking_lists_with_rank.merge(
        res_df_scores, 
        on=['qid', 'docno'], 
        how='left'
    )
    
    # Identify records with missing scores
    missing_scores = merged_df[merged_df['score'].isna()]
    
    if len(missing_scores) > 0:
        retriever = OptimizedDocumentRetriever(index)
        
        for idx, row in missing_scores.iterrows():
            qid = row['qid']
            docno = row['docno']
            query_text = queries_df[queries_df['qid'] == qid]['query'].iloc[0]
            
            try:
                score = retriever.lmjm_score(str(docno), query_text)
            except:
                score = 0.001
            
            merged_df.at[idx, 'score'] = score
    if file_name is not None:
        merged_df.to_csv(file_name, index=False)
    else:
        merged_df.to_csv('merged_df.csv', index=False)
    return merged_df

# RBO method calculation
def compute_rbo(list1, list2, p=constants.RBO_P):
    """
    Compute RBO similarity (finite prefix, non tie-aware)
    :param list1: Ranking list 1
    :param list2: Ranking list 2
    :param p: Persistence parameter (0<p<1), larger values focus more on deeper positions
    """
    if not list1 or not list2:
        return 0.0
    if not (0 < p < 1):
        raise ValueError("p must be in the range (0, 1)")

    k = max(len(list1), len(list2))
    seen1, seen2 = set(), set()
    weighted_sum = 0.0
    overlap = 0  # X_d

    for d in range(1, k + 1):
        if d <= len(list1):
            seen1.add(list1[d - 1])
        if d <= len(list2):
            seen2.add(list2[d - 1])

        overlap = len(seen1.intersection(seen2))
        A_d = overlap / d
        weighted_sum += A_d * (p ** d)

    rbo_value = ((1 - p) / p) * weighted_sum
    return rbo_value

# Get grouped lists based on single qid and given seq_no list, return as dictionary
def get_group_lists(complete_df, qid, seq_no_list):
    """
    Get corresponding DataFrame groups based on query ID and sequence number list
    
    :param complete_df: Complete DataFrame
    :param qid: Query ID
    :param seq_no_list: Sequence number list
    :return: Dictionary in {seq_no: df} format
    """
    result = {}
    # Filter data for specified qid
    qid_df = complete_df[complete_df['qid'] == qid]

    # Get corresponding DataFrame based on seq_no_list
    for seq_no in seq_no_list:
        seq_df = qid_df[qid_df['seq_no'] == seq_no]
        if not seq_df.empty:
            result[seq_no] = seq_df
    
    return result

# NQC normalization method calculation
def nqc_cv_no_train(scores, eps=1e-8):
    mu = float(np.mean(scores))
    sd = float(np.std(scores, ddof=0))
    cv = sd / (abs(mu) + eps)
    return cv / (1.0 + cv)  # or 1 - np.exp(-cv)

# Compute single group RBO-QPP method
def compute_singel_rbo_qpp(query_dict, 
                           group_lists, 
                           nqc_specificity, 
                           cutoff = None, 
                           use_normalization=False, 
                           uncertainty = False,
                           normalization_method='sigmoid'
                           ):
    """
    Compute RBO-based QPP score for a single query (group processing)
    
    Algorithm flow:
    1. Get all ranking lists for the query, group by group_size
    2. Compute RBO-QPP for each group separately:
        - Calculate pairwise RBO similarities within group, find "consensus ranking"
        - Calculate RBO score between each list and consensus ranking as weight
        - Compute NQC score for each list, weighted average using RBO weights
    3. Aggregate results from all groups
    
    :param query_dict: Dictionary of query ID and query text
    :param group_lists: Dictionary of grouped lists
    :param nqc_specificity: NQCSpecificity object
    :param uef_specificity: UEFSpecificity object
    :param rsd_specificity: RSDSpecificity object
    :param cutoff: Cutoff position
    :param use_normalization: Whether to apply normalization to NQC scores
    :param uncertainty: Whether to compute uncertainty
    :param normalization_method: Normalization method ('cv_replace', 'cv_multiply', 'sigmoid', 'tanh', 'minmax')
    :return: Final RBO-QPP score
    """
    beta = 0.01
    qid = list(query_dict.keys())[0]
    query = query_dict[qid]
    
    # 1. Calculate pairwise RBO similarities within group, find "consensus ranking"
    doc_lists = []
    seq_nos = list(group_lists.keys())
    
    # Extract document list for each seq_no
    for seq_no in seq_nos:
        if cutoff is not None:
            df = group_lists[seq_no].head(cutoff)
        else:
            df = group_lists[seq_no]
        docs = df['docno'].tolist()
        doc_lists.append(docs)
    
    if len(doc_lists) < 2:
        return {'error': f'Insufficient lists in group for query {qid} (at least 2 lists required)'}
    
    # 2. Calculate pairwise RBO similarity matrix
    n = len(doc_lists)
    rbo_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            rbo_sim = compute_rbo(doc_lists[i], doc_lists[j], constants.RBO_P)
            rbo_matrix[i][j] = rbo_sim
            rbo_matrix[j][i] = rbo_sim
        rbo_matrix[i][i] = 1.0  # Self-similarity is 1
    
    # 3. Find list with highest average RBO similarity as "consensus ranking"
    avg_similarities = []
    for i in range(n):
        avg_sim = sum(rbo_matrix[i]) / n
        avg_similarities.append(avg_sim)
    
    consensus_idx = avg_similarities.index(max(avg_similarities))
    consensus_ranking = doc_lists[consensus_idx]
    
    # print(f"    Selected list {consensus_idx} as consensus ranking, average similarity: {max(avg_similarities):.4f}")

    # 4. Calculate RBO score between each list and consensus ranking as weight
    weights = []
    for i in range(n):
        weight = compute_rbo(doc_lists[i], consensus_ranking)
        weights.append(weight)
    
    # print("RBO scores between each list and consensus ranking as weights", weights)
    
    # 5. Total weight value
    weights_sum = sum(weights)
    # print("Total weight value", weights_sum)
    # 6. Compute NQC scores for multiple lists, weighted average using RBO weights
    nqc_scores = []
    for i in range(n):
        nqc_score = nqc_specificity.compute_specificity(query, group_lists[seq_nos[i]], cutoff)
        
        # Apply normalization here (optional)
        if use_normalization:
            # Force apply normalization (regardless of whether NQC object supports it)
            normalized_nqc_score = apply_normalization(group_lists[seq_nos[i]], nqc_score, cutoff=cutoff, method=normalization_method)
            nqc_score = normalized_nqc_score

        # print(f"Normalized RBO-QPP score for list {i+1} of query {qid}: {weights[i]*nqc_score/weights_sum}")
        # print("--------------------------------")
        nqc_scores.append(weights[i]*nqc_score/weights_sum)
    # print("Normalized RBO-QPP score for each list", nqc_scores)
    # print("Final RBO-QPP score", sum(nqc_scores))
    # 
    final_qpp = sum(nqc_scores)
    if uncertainty:
        uncertainty = np.var(weights) if len(weights) > 1 else 0.0
        final_qpp = final_qpp - beta * uncertainty
    return final_qpp

def apply_normalization(results_df, nqc_score, cutoff=None, eps=1e-8, method='sigmoid'):
    """
    Apply normalization to NQC scores
    
    Args:
        results_df: Search results DataFrame
        nqc_score: Original NQC score
        cutoff: Cutoff position
        eps: Small value to prevent division by zero
        method: Normalization method ('cv_replace', 'cv_multiply', 'sigmoid', 'tanh')
        
    Returns:
        Normalized NQC score
    """
    if results_df.empty:
        return nqc_score
    
    # Get scores
    if cutoff is not None:
        scores = results_df.head(cutoff)['score'].values
    else:
        scores = results_df['score'].values
    
    if len(scores) <= 1:
        return nqc_score
    
    if method == 'cv_replace':
        # Method 1: Directly replace NQC score with CV value (ensure in [0,1) range)
        return nqc_cv_no_train(scores, eps)
    
    elif method == 'cv_multiply':
        # Method 2: Original method (may exceed 1)
        return nqc_cv_no_train(scores, eps) * nqc_score
    
    elif method == 'sigmoid':
        # Method 3: Use sigmoid function to normalize to (0,1)
        cv_factor = nqc_cv_no_train(scores, eps)
        adjusted_nqc = cv_factor * nqc_score
        return 1 / (1 + np.exp(-adjusted_nqc))  # sigmoid normalization
    
    elif method == 'tanh':
        # Method 4: Use tanh function to normalize to (-1,1), then map to (0,1)
        cv_factor = nqc_cv_no_train(scores, eps)
        adjusted_nqc = cv_factor * nqc_score
        return (np.tanh(adjusted_nqc) + 1) / 2  # tanh normalization and map to [0,1]
    
    elif method == 'minmax':
        # Method 5: Min-Max normalization (requires preset maximum value)
        cv_factor = nqc_cv_no_train(scores, eps)
        adjusted_nqc = cv_factor * nqc_score
        # Assume reasonable NQC maximum value is 10 (can be adjusted based on actual situation)
        max_expected_nqc = 10.0
        return min(adjusted_nqc / max_expected_nqc, 1.0)
    
    else:
        return nqc_score

# Experiment process
def experiment_process(queries_df, 
                       complete_df, 
                       group_size, 
                       nqc_specificity,
                       uef_specificity,
                       rsd_specificity,
                       use_normalization=False, 
                       normalization_method='sigmoid',
                       uncertainty=False,
                       test_name=None):
    seq_no_list = complete_df['seq_no'].unique()
    if len(seq_no_list) < group_size:
        raise ValueError(f"Number of query lists is less than {group_size}, cannot perform grouping")
    
    # If not divisible, take maximum number of groups, e.g., 100/3 takes 33 groups
    # if len(seq_no_list) % group_size != 0:
    #     raise ValueError(f"Number of query lists cannot be divided by {group_size}, cannot perform grouping")
    
    # 1. Get grouping settings
    group_num = len(seq_no_list) // group_size
    all_group_lists = []
    for i in range(group_num):
        group_lists = []
        for j in range(group_size):
            group_lists.append(seq_no_list[i*group_size+j])
        all_group_lists.append(group_lists)

    
    # 2. Loop through qids for experiments, calculate IR metrics and QPP method metrics separately
    qids = queries_df['qid'].unique()
    # qids = qids[:3]
    evaluator = Evaluator(constants.QRELS_TREC_FAIR, {})

    # **Optimization 1: Pre-build query dictionary to avoid repeated lookups**
    query_lookup = {}
    for _, row in queries_df.iterrows():
        query_lookup[row['qid']] = row['query']

    # Store detailed data
    detailed_results = []
    
    # Group-based metrics - grouped by query
    all_qids_group_ndcg_values = []
    all_qids_group_nqc_values = []
    all_qids_group_rbo_nqc_qpp_values = []
    all_qids_group_rbo_uef_qpp_values = []
    all_qids_group_rbo_rsd_qpp_values = []

    # Non-group calculation, with single list as unit, calculate IR metrics and QPP method metrics
    none_group_ndcg_values = []
    none_group_NQC_values = []
    none_group_UEF_values = []
    none_group_RSD_values = []
    
    for i, qid in enumerate(qids):
        time_start = time.time()
        print(f"Starting calculation of metrics for query {qid}, {len(qids)-i-1} queries remaining")
        # All group metrics for each qid
        single_qid_group_ndcg_values = []
        single_qid_group_nqc_values = []
        single_qid_group_rbo_nqc_qpp_values = []
        single_qid_group_rbo_uef_qpp_values = []
        single_qid_group_rbo_rsd_qpp_values = []
        # **Optimization 2: Directly use pre-built query dictionary**
        qid_query_dict = {qid: query_lookup[qid]}
        
        # Loop through all lists in one group here
        for group_idx, group_lists in enumerate(all_group_lists):
            group_lists_dict = get_group_lists(complete_df, qid, group_lists)
            # IR values and traditional QPP values for each group
            single_group_ndcg_values = []
            single_group_nqc_values = []
            
            for seq_no in group_lists:
                if seq_no not in group_lists_dict:
                    print(f"Warning: seq_no {seq_no} not found in query {qid}")
                    continue
                    
                # 2.1 Calculate IR metrics for each single list in each group
                evaluator.top_docs_map = {qid: group_lists_dict[seq_no]}
                ndcg = evaluator.compute(qid, 'nDCG', 10)
                single_group_ndcg_values.append(ndcg)
                none_group_ndcg_values.append(ndcg)
                
                # 2.2 Calculate QPP method metrics for each single list in each group
                nqc_score = nqc_specificity.compute_specificity(query_lookup[qid], group_lists_dict[seq_no])
                # Normalization
                if use_normalization:
                    nqc_score = apply_normalization(group_lists_dict[seq_no], nqc_score, method=normalization_method)
                single_group_nqc_values.append(nqc_score)
                none_group_NQC_values.append(nqc_score)
                
                # Calculate UEF score
                uef_score = uef_specificity.compute_specificity(query_lookup[qid], group_lists_dict[seq_no])
                if use_normalization:
                    uef_score = apply_normalization(group_lists_dict[seq_no], uef_score, method=normalization_method)
                none_group_UEF_values.append(uef_score)
                
                # Calculate RSD score
                rsd_score = rsd_specificity.compute_specificity(query_lookup[qid], group_lists_dict[seq_no])
                if use_normalization:
                    rsd_score = apply_normalization(group_lists_dict[seq_no], rsd_score, method=normalization_method)
                none_group_RSD_values.append(rsd_score)
                # **Record detailed data for CSV output**
                detailed_results.append({
                    'qid': qid,
                    'group_idx': group_idx,
                    'seq_no': seq_no,
                    'ndcg': ndcg,
                    'nqc': nqc_score,
                    'uef': uef_score,
                    'rsd': rsd_score
                })

            # 2.3 Calculate RBO-QPP metrics for each group
            if len(single_group_ndcg_values) > 0:  # Ensure there is data
                final_qpp = compute_singel_rbo_qpp(qid_query_dict, group_lists_dict, nqc_specificity,
                                                  use_normalization=use_normalization,
                                                  uncertainty=uncertainty,
                                                  normalization_method=normalization_method)
                final_uef = compute_singel_rbo_qpp(qid_query_dict, group_lists_dict, uef_specificity,
                                                  use_normalization=use_normalization,
                                                  uncertainty=uncertainty,
                                                  normalization_method=normalization_method)
                final_rsd = compute_singel_rbo_qpp(qid_query_dict, group_lists_dict, rsd_specificity,
                                                  use_normalization=use_normalization,
                                                  uncertainty=uncertainty,
                                                  normalization_method=normalization_method)

                single_qid_group_ndcg_values.append(np.mean(single_group_ndcg_values))
                single_qid_group_nqc_values.append(np.mean(single_group_nqc_values))
                single_qid_group_rbo_nqc_qpp_values.append(final_qpp)
                single_qid_group_rbo_uef_qpp_values.append(final_uef)
                single_qid_group_rbo_rsd_qpp_values.append(final_rsd)
                # **Update group-level data in detailed results**
                for result in detailed_results:
                    if result['qid'] == qid and result['group_idx'] == group_idx:
                        result['rbo_nqc_qpp_group'] = final_qpp
                        result['rbo_uef_qpp_group'] = final_uef
                        result['rbo_rsd_qpp_group'] = final_rsd
                        result['ndcg_group_avg'] = np.mean(single_group_ndcg_values)
                        result['nqc_group_avg'] = np.mean(single_group_nqc_values)
                        
                        # Note: UEF and RSD have already been added in the loop above
        # Group metrics for each qid
        print(f"Query {qid}: NDCG groups={len(single_qid_group_ndcg_values)}, NQC groups={len(single_qid_group_nqc_values)}, RBO-NQC-QPP groups={len(single_qid_group_rbo_nqc_qpp_values)}, RBO-UEF-QPP groups={len(single_qid_group_rbo_uef_qpp_values)}, RBO-RSD-QPP groups={len(single_qid_group_rbo_rsd_qpp_values)}")
        print(f"  NDCG values: {[f'{v:.4f}' for v in single_qid_group_ndcg_values]}")
        print(f"  NQC values:  {[f'{v:.4f}' for v in single_qid_group_nqc_values]}")
        print(f"  RBO-NQC-QPP values:  {[f'{v:.4f}' for v in single_qid_group_rbo_nqc_qpp_values]}")
        print(f"  RBO-UEF-QPP values:  {[f'{v:.4f}' for v in single_qid_group_rbo_uef_qpp_values]}")
        print(f"  RBO-RSD-QPP values:  {[f'{v:.4f}' for v in single_qid_group_rbo_rsd_qpp_values]}")
        print(f"  Total calculations: NDCG({len(none_group_ndcg_values)}), NQC({len(none_group_NQC_values)}), UEF({len(none_group_UEF_values)}), RSD({len(none_group_RSD_values)})")
        # print(f"Non-group NQC values: {none_group_NQC_values}")
        # print(f"Non-group UEF values: {none_group_UEF_values}")
        # print(f"Non-group RSD values: {none_group_RSD_values}")
        all_qids_group_ndcg_values.append(single_qid_group_ndcg_values)
        all_qids_group_nqc_values.append(single_qid_group_nqc_values)
        all_qids_group_rbo_nqc_qpp_values.append(single_qid_group_rbo_nqc_qpp_values)
        all_qids_group_rbo_uef_qpp_values.append(single_qid_group_rbo_uef_qpp_values)
        all_qids_group_rbo_rsd_qpp_values.append(single_qid_group_rbo_rsd_qpp_values)
        print('--------------------------------')
        time_end = time.time()
        print(f"Metric calculation time for query {qid}: {time_end - time_start:.2f} seconds")
        print('--------------------------------')
    # **Output detailed data to CSV**
    detailed_df = pd.DataFrame(detailed_results)
    if test_name is not None:
        if use_normalization and uncertainty == False:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_detailed_metrics_analysis_{timestamp}_with_normalization_{normalization_method}.csv', index=False)
        elif uncertainty and use_normalization == False:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_detailed_metrics_analysis_{timestamp}_with_uncertainty.csv', index=False)
        elif use_normalization and uncertainty:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_detailed_metrics_analysis_{timestamp}_with_normalization_{normalization_method}_with_uncertainty.csv', index=False)
        else:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_detailed_metrics_analysis_{timestamp}.csv', index=False)
    else:
        if use_normalization and uncertainty == False:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_detailed_metrics_analysis_{timestamp}_with_normalization_{normalization_method}.csv', index=False)
        elif uncertainty and use_normalization == False:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_detailed_metrics_analysis_{timestamp}_with_uncertainty.csv', index=False)
        elif use_normalization and uncertainty:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_detailed_metrics_analysis_{timestamp}_with_normalization_{normalization_method}_with_uncertainty.csv', index=False)
        else:
            detailed_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_detailed_metrics_analysis_{timestamp}.csv', index=False)
    # print(f"Detailed metric data saved to group_size_{group_size}_detailed_metrics_analysis_{timestamp}.csv")
    
    # print("--------------------------------")
    # print(f"Total queries: {len(all_qids_group_ndcg_values)}")
    # if len(all_qids_group_ndcg_values) > 0:
    #     print(f"Groups per query: {len(all_qids_group_ndcg_values[0])}")

    # **Fix 3: Correct tau calculation logic - calculate tau by group**
    group_tau_results = []
    
    # Iterate through all groups
    for group_idx in range(len(all_qids_group_ndcg_values[0]) if all_qids_group_ndcg_values else 0):
        # Collect NDCG, NQC and RBO-QPP values for this group across all queries
        group_ndcg_values = []
        group_nqc_values = []
        group_rbo_nqc_qpp_values = []
        group_rbo_uef_qpp_values = []
        group_rbo_rsd_qpp_values = []
        
        for qid_idx, qid in enumerate(qids):
            if (qid_idx < len(all_qids_group_ndcg_values) and 
                group_idx < len(all_qids_group_ndcg_values[qid_idx])):
                group_ndcg_values.append(all_qids_group_ndcg_values[qid_idx][group_idx])
                group_nqc_values.append(all_qids_group_nqc_values[qid_idx][group_idx])
                group_rbo_nqc_qpp_values.append(all_qids_group_rbo_nqc_qpp_values[qid_idx][group_idx])
                group_rbo_uef_qpp_values.append(all_qids_group_rbo_uef_qpp_values[qid_idx][group_idx])
                group_rbo_rsd_qpp_values.append(all_qids_group_rbo_rsd_qpp_values[qid_idx][group_idx])
        
        if len(group_ndcg_values) > 1:  # At least 2 data points needed to calculate correlation
            nqc_tau, nqc_p = kendalltau(group_ndcg_values, group_nqc_values)
            rbo_nqc_qpp_tau, rbo_nqc_qpp_p = kendalltau(group_ndcg_values, group_rbo_nqc_qpp_values)
            rbo_uef_qpp_tau, rbo_uef_qpp_p = kendalltau(group_ndcg_values, group_rbo_uef_qpp_values)
            rbo_rsd_qpp_tau, rbo_rsd_qpp_p = kendalltau(group_ndcg_values, group_rbo_rsd_qpp_values)
            
            group_tau_results.append({
                'group_idx': group_idx,
                'nqc_tau': nqc_tau,
                'nqc_p_value': nqc_p,
                'rbo_nqc_qpp_tau': rbo_nqc_qpp_tau,
                'rbo_nqc_qpp_p_value': rbo_nqc_qpp_p,
                'rbo_uef_qpp_tau': rbo_uef_qpp_tau,
                'rbo_uef_qpp_p_value': rbo_uef_qpp_p,
                'rbo_rsd_qpp_tau': rbo_rsd_qpp_tau,
                'rbo_rsd_qpp_p_value': rbo_rsd_qpp_p,
                'ndcg_values': group_ndcg_values,
                'nqc_values': group_nqc_values,
                'rbo_nqc_qpp_values': group_rbo_nqc_qpp_values,
                'rbo_uef_qpp_values': group_rbo_uef_qpp_values,
                'rbo_rsd_qpp_values': group_rbo_rsd_qpp_values
            })
            
            print(f"Group {group_idx}: NDCG vs NQC tau={nqc_tau:.4f}(p={nqc_p:.4f}), NDCG vs RBO-NQC-QPP tau={rbo_nqc_qpp_tau:.4f}(p={rbo_nqc_qpp_p:.4f}), NDCG vs RBO-UEF-QPP tau={rbo_uef_qpp_tau:.4f}(p={rbo_uef_qpp_p:.4f}), NDCG vs RBO-RSD-QPP tau={rbo_rsd_qpp_tau:.4f}(p={rbo_rsd_qpp_p:.4f})")
            # print(f"  Data points: {len(group_ndcg_values)}")
        else:
            pass
            # print(f"Group {group_idx}: Insufficient data points (only {len(group_ndcg_values)} queries), cannot calculate tau")
        print('--------------------------------')
    
    # **Reorganize data: group by seq_no (single list approach)**
    # Reorganize data by seq_no into nested structure
    seq_no_groups = {}  # {seq_no: {'ndcg': [values], 'nqc': [values], 'uef': [values], 'rsd': [values]}}
    
    # Re-extract data from detailed_results
    for result in detailed_results:
        seq_no = result['seq_no']
        if seq_no not in seq_no_groups:
            seq_no_groups[seq_no] = {'ndcg': [], 'nqc': [], 'uef': [], 'rsd': []}
        
        seq_no_groups[seq_no]['ndcg'].append(result['ndcg'])
        seq_no_groups[seq_no]['nqc'].append(result['nqc'])
        seq_no_groups[seq_no]['uef'].append(result['uef'])
        seq_no_groups[seq_no]['rsd'].append(result['rsd'])
    
    print(f"Data reorganized by seq_no: total {len(seq_no_groups)} seq_nos")
    
    # **Calculate single list tau analysis (each seq_no as a group)**
    single_list_tau_results = []
    
    for seq_no, values in seq_no_groups.items():
        if len(values['ndcg']) > 1:  # At least 2 data points needed
            # Calculate tau between each method and NDCG
            nqc_tau, nqc_p = kendalltau(values['ndcg'], values['nqc'])
            uef_tau, uef_p = kendalltau(values['ndcg'], values['uef']) 
            rsd_tau, rsd_p = kendalltau(values['ndcg'], values['rsd'])
            
            single_list_tau_results.append({
                'seq_no': seq_no,
                'nqc_tau': nqc_tau,
                'nqc_p_value': nqc_p,
                'uef_tau': uef_tau,
                'uef_p_value': uef_p,
                'rsd_tau': rsd_tau,
                'rsd_p_value': rsd_p,
                'data_points': len(values['ndcg'])
            })
    
    print(f"Single list tau analysis: calculated tau for {len(single_list_tau_results)} seq_nos")
    
    # **Calculate single list delta tau (same logic as group)**
    nqc_single_delta_tau_values = []
    uef_single_delta_tau_values = []
    rsd_single_delta_tau_values = []
    
    # Calculate single list delta tau
    for i in range(len(single_list_tau_results)):
        # NQC delta tau
        nqc_anchor_tau = single_list_tau_results[i]['nqc_tau']
        # Check if anchor_tau is nan
        if not np.isnan(nqc_anchor_tau):
            nqc_others_tau = [single_list_tau_results[j]['nqc_tau'] for j in range(len(single_list_tau_results)) 
                             if j != i and not np.isnan(single_list_tau_results[j]['nqc_tau'])]
            if len(nqc_others_tau) > 0:
                nqc_avg_tau_others = np.mean(nqc_others_tau)
                nqc_delta_tau_val = abs(nqc_anchor_tau - nqc_avg_tau_others)
                nqc_single_delta_tau_values.append(nqc_delta_tau_val)
        
        # UEF delta tau
        uef_anchor_tau = single_list_tau_results[i]['uef_tau']
        # Check if anchor_tau is nan
        if not np.isnan(uef_anchor_tau):
            uef_others_tau = [single_list_tau_results[j]['uef_tau'] for j in range(len(single_list_tau_results)) 
                             if j != i and not np.isnan(single_list_tau_results[j]['uef_tau'])]
            if len(uef_others_tau) > 0:
                uef_avg_tau_others = np.mean(uef_others_tau)
                uef_delta_tau_val = abs(uef_anchor_tau - uef_avg_tau_others)
                uef_single_delta_tau_values.append(uef_delta_tau_val)
        
        # RSD delta tau
        rsd_anchor_tau = single_list_tau_results[i]['rsd_tau']
        # Check if anchor_tau is nan
        if not np.isnan(rsd_anchor_tau):
            rsd_others_tau = [single_list_tau_results[j]['rsd_tau'] for j in range(len(single_list_tau_results)) 
                             if j != i and not np.isnan(single_list_tau_results[j]['rsd_tau'])]
            if len(rsd_others_tau) > 0:
                rsd_avg_tau_others = np.mean(rsd_others_tau)
                rsd_delta_tau_val = abs(rsd_anchor_tau - rsd_avg_tau_others)
                rsd_single_delta_tau_values.append(rsd_delta_tau_val)
    
    # Calculate average tau and delta tau
    valid_nqc_single_tau = [r['nqc_tau'] for r in single_list_tau_results if not np.isnan(r['nqc_tau'])]
    valid_uef_single_tau = [r['uef_tau'] for r in single_list_tau_results if not np.isnan(r['uef_tau'])]
    valid_rsd_single_tau = [r['rsd_tau'] for r in single_list_tau_results if not np.isnan(r['rsd_tau'])]
    
    if valid_nqc_single_tau:
        nqc_single_tau = np.mean(valid_nqc_single_tau)
        print(f"Single list NQC average tau: {nqc_single_tau:.4f} (based on {len(valid_nqc_single_tau)} seq_nos)")
    else:
        nqc_single_tau = 0.0
        
    if valid_uef_single_tau:
        uef_single_tau = np.mean(valid_uef_single_tau)
        print(f"Single list UEF average tau: {uef_single_tau:.4f} (based on {len(valid_uef_single_tau)} seq_nos)")
    else:
        uef_single_tau = 0.0
        
    if valid_rsd_single_tau:
        rsd_single_tau = np.mean(valid_rsd_single_tau)
        print(f"Single list RSD average tau: {rsd_single_tau:.4f} (based on {len(valid_rsd_single_tau)} lists)")
    else:
        rsd_single_tau = 0.0
    
    # Print delta tau results
    if nqc_single_delta_tau_values:
        print(f"Single list NQC average delta tau: {np.mean(nqc_single_delta_tau_values):.4f}")
    if uef_single_delta_tau_values:
        print(f"Single list UEF average delta tau: {np.mean(uef_single_delta_tau_values):.4f}")
    if rsd_single_delta_tau_values:
        print(f"Single list RSD average delta tau: {np.mean(rsd_single_delta_tau_values):.4f}")
    
    # **Output tau analysis results to CSV**
    if group_tau_results:
        tau_df = pd.DataFrame([{
            'group_idx': result['group_idx'],
            'nqc_tau': result['nqc_tau'],
            'rbo_nqc_qpp_tau': result['rbo_nqc_qpp_tau'],
            'rbo_uef_qpp_tau': result['rbo_uef_qpp_tau'],
            'rbo_rsd_qpp_tau': result['rbo_rsd_qpp_tau'],
        } for result in group_tau_results])
        if test_name is not None:
            if use_normalization and uncertainty == False:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_tau_analysis_by_group_{timestamp}_with_normalization_{normalization_method}.csv', index=False)
            elif uncertainty and use_normalization == False:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_tau_analysis_by_group_{timestamp}_with_uncertainty.csv', index=False)
            elif use_normalization and uncertainty:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_tau_analysis_by_group_{timestamp}_with_normalization_{normalization_method}_with_uncertainty.csv', index=False)
            else:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_tau_analysis_by_group_{timestamp}.csv', index=False)
        else:
            if use_normalization and uncertainty == False:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_tau_analysis_by_group_{timestamp}_with_normalization_{normalization_method}.csv', index=False)
            elif uncertainty and use_normalization == False:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_tau_analysis_by_group_{timestamp}_with_uncertainty.csv', index=False)
            elif use_normalization and uncertainty:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_tau_analysis_by_group_{timestamp}_with_normalization_{normalization_method}_with_uncertainty.csv', index=False)
            else:
                tau_df.to_csv(f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_tau_analysis_by_group_{timestamp}.csv', index=False)
        print(f"Tau analysis results saved to {constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_tau_analysis_by_group_{timestamp}.csv")
        
        # Calculate average tau
        valid_nqc_tau = [r['nqc_tau'] for r in group_tau_results if not np.isnan(r['nqc_tau'])]
        valid_rbo_nqc_qpp_tau = [r['rbo_nqc_qpp_tau'] for r in group_tau_results if not np.isnan(r['rbo_nqc_qpp_tau'])]
        valid_rbo_uef_qpp_tau = [r['rbo_uef_qpp_tau'] for r in group_tau_results if not np.isnan(r['rbo_uef_qpp_tau'])]
        valid_rbo_rsd_qpp_tau = [r['rbo_rsd_qpp_tau'] for r in group_tau_results if not np.isnan(r['rbo_rsd_qpp_tau'])]
        
        if valid_nqc_tau:
            print(f'Average NQC_group tau: {np.mean(valid_nqc_tau):.4f} (std: {np.std(valid_nqc_tau):.4f})')
        if valid_rbo_nqc_qpp_tau:
            print(f'Average RBO-NQC-QPP tau: {np.mean(valid_rbo_nqc_qpp_tau):.4f} (std: {np.std(valid_rbo_nqc_qpp_tau):.4f})')
        if valid_rbo_uef_qpp_tau:
            print(f'Average RBO-UEF-QPP tau: {np.mean(valid_rbo_uef_qpp_tau):.4f} (std: {np.std(valid_rbo_uef_qpp_tau):.4f})')
        if valid_rbo_rsd_qpp_tau:
            print(f'Average RBO-RSD-QPP tau: {np.mean(valid_rbo_rsd_qpp_tau):.4f} (std: {np.std(valid_rbo_rsd_qpp_tau):.4f})')
        

        
        # **Calculate delta tau - by group**
        print("\n=== Delta Tau Calculation ===")
        nqc_delta_tau_values = []
        rbo_nqc_qpp_delta_tau_values = []
        rbo_uef_qpp_delta_tau_values = []
        rbo_rsd_qpp_delta_tau_values = []
        
        # Calculate delta tau for both NQC and RBO-QPP simultaneously
        for i in range(len(group_tau_results)):
            # Calculate NQC delta tau
            nqc_anchor_tau = group_tau_results[i]['nqc_tau']
            rbo_nqc_qpp_anchor_tau = group_tau_results[i]['rbo_nqc_qpp_tau']
            rbo_uef_qpp_anchor_tau = group_tau_results[i]['rbo_uef_qpp_tau']
            rbo_rsd_qpp_anchor_tau = group_tau_results[i]['rbo_rsd_qpp_tau']
            
            nqc_others_tau = [group_tau_results[j]['nqc_tau'] for j in range(len(group_tau_results)) if j != i and not np.isnan(group_tau_results[j]['nqc_tau'])]
            rbo_nqc_qpp_others_tau = [group_tau_results[j]['rbo_nqc_qpp_tau'] for j in range(len(group_tau_results)) if j != i and not np.isnan(group_tau_results[j]['rbo_nqc_qpp_tau'])]
            rbo_uef_qpp_others_tau = [group_tau_results[j]['rbo_uef_qpp_tau'] for j in range(len(group_tau_results)) if j != i and not np.isnan(group_tau_results[j]['rbo_uef_qpp_tau'])]
            rbo_rsd_qpp_others_tau = [group_tau_results[j]['rbo_rsd_qpp_tau'] for j in range(len(group_tau_results)) if j != i and not np.isnan(group_tau_results[j]['rbo_rsd_qpp_tau'])]
            

            if len(nqc_others_tau) > 0:
                nqc_avg_tau_others = np.mean(nqc_others_tau)
                nqc_delta_tau_val = abs(nqc_anchor_tau - nqc_avg_tau_others)
                nqc_delta_tau_values.append(nqc_delta_tau_val)

            if len(rbo_nqc_qpp_others_tau) > 0:
                rbo_nqc_qpp_avg_tau_others = np.mean(rbo_nqc_qpp_others_tau)
                rbo_nqc_qpp_delta_tau_val = abs(rbo_nqc_qpp_anchor_tau - rbo_nqc_qpp_avg_tau_others)
                rbo_nqc_qpp_delta_tau_values.append(rbo_nqc_qpp_delta_tau_val)

            if len(rbo_uef_qpp_others_tau) > 0:
                rbo_uef_qpp_avg_tau_others = np.mean(rbo_uef_qpp_others_tau)
                rbo_uef_qpp_delta_tau_val = abs(rbo_uef_qpp_anchor_tau - rbo_uef_qpp_avg_tau_others)
                rbo_uef_qpp_delta_tau_values.append(rbo_uef_qpp_delta_tau_val)

            if len(rbo_rsd_qpp_others_tau) > 0:
                rbo_rsd_qpp_avg_tau_others = np.mean(rbo_rsd_qpp_others_tau)
                rbo_rsd_qpp_delta_tau_val = abs(rbo_rsd_qpp_anchor_tau - rbo_rsd_qpp_avg_tau_others)
                rbo_rsd_qpp_delta_tau_values.append(rbo_rsd_qpp_delta_tau_val)
        
        if nqc_delta_tau_values:
            print(f'\nAverage NQC_group delta tau: {np.mean(nqc_delta_tau_values):.4f} (std: {np.std(nqc_delta_tau_values):.4f})')
        if rbo_nqc_qpp_delta_tau_values:
            print(f'Average RBO-NQC-QPP delta tau: {np.mean(rbo_nqc_qpp_delta_tau_values):.4f} (std: {np.std(rbo_nqc_qpp_delta_tau_values):.4f})')
        if rbo_uef_qpp_delta_tau_values:
            print(f'Average RBO-UEF-QPP delta tau: {np.mean(rbo_uef_qpp_delta_tau_values):.4f} (std: {np.std(rbo_uef_qpp_delta_tau_values):.4f})')
        if rbo_rsd_qpp_delta_tau_values:
            print(f'Average RBO-RSD-QPP delta tau: {np.mean(rbo_rsd_qpp_delta_tau_values):.4f} (std: {np.std(rbo_rsd_qpp_delta_tau_values):.4f})')
        
        # Print single list delta tau
        if nqc_single_delta_tau_values:
            print(f'Average NQC_single delta tau: {np.mean(nqc_single_delta_tau_values):.4f} (std: {np.std(nqc_single_delta_tau_values):.4f})')
        if uef_single_delta_tau_values:
            print(f'Average UEF delta tau: {np.mean(uef_single_delta_tau_values):.4f} (std: {np.std(uef_single_delta_tau_values):.4f})')
        if rsd_single_delta_tau_values:
            print(f'Average RSD delta tau: {np.mean(rsd_single_delta_tau_values):.4f} (std: {np.std(rsd_single_delta_tau_values):.4f})')
        
        # **Build method comparison CSV output**
        methods_summary = []
        
        # NQC method
        if valid_nqc_tau:
            methods_summary.append({
                'Method': 'NQC',
                'tau': nqc_single_tau,
                'delta_tau': np.mean(nqc_single_delta_tau_values) if nqc_single_delta_tau_values else 0.0
            })
        
        
        # UEF method (based on single list tau)
        methods_summary.append({
            'Method': 'UEF',
            'tau': uef_single_tau,
            'delta_tau': np.mean(uef_single_delta_tau_values) if uef_single_delta_tau_values else 0.0
        })
        
        # RSD method (based on single list tau)
        methods_summary.append({
            'Method': 'RSD',
            'tau': rsd_single_tau,
            'delta_tau': np.mean(rsd_single_delta_tau_values) if rsd_single_delta_tau_values else 0.0
        })
        
        # RBO-NQC-QPP method
        if valid_rbo_nqc_qpp_tau:
            methods_summary.append({
                'Method': 'RBO-NQC-QPP',
                'tau': np.mean(valid_rbo_nqc_qpp_tau),
                'delta_tau': np.mean(rbo_nqc_qpp_delta_tau_values) if rbo_nqc_qpp_delta_tau_values else 0.0
            })
        
        # RBO-UEF-QPP method
        if valid_rbo_uef_qpp_tau:
            methods_summary.append({
                'Method': 'RBO-UEF-QPP',
                'tau': np.mean(valid_rbo_uef_qpp_tau),
                'delta_tau': np.mean(rbo_uef_qpp_delta_tau_values) if rbo_uef_qpp_delta_tau_values else 0.0
            })
        
        # RBO-RSD-QPP method
        if valid_rbo_rsd_qpp_tau:
            methods_summary.append({
                'Method': 'RBO-RSD-QPP',
                'tau': np.mean(valid_rbo_rsd_qpp_tau),
                'delta_tau': np.mean(rbo_rsd_qpp_delta_tau_values) if rbo_rsd_qpp_delta_tau_values else 0.0
            })
        
        # Create method comparison CSV
        if methods_summary:
            methods_df = pd.DataFrame(methods_summary)
            if test_name is not None:
                if use_normalization and uncertainty == False:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_methods_comparison_{timestamp}_with_normalization_{normalization_method}.csv'
                elif uncertainty and use_normalization == False:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_methods_comparison_{timestamp}_with_uncertainty.csv'
                elif use_normalization and uncertainty:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_methods_comparison_{timestamp}_with_normalization_{normalization_method}_with_uncertainty.csv'
                else:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/{test_name}_group_size_{group_size}_methods_comparison_{timestamp}.csv'
            else:
                if use_normalization and uncertainty == False:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_methods_comparison_{timestamp}_with_normalization_{normalization_method}.csv'
                elif uncertainty and use_normalization == False:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_methods_comparison_{timestamp}_with_uncertainty.csv'
                elif use_normalization and uncertainty:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_methods_comparison_{timestamp}_with_normalization_{normalization_method}_with_uncertainty.csv'
                else:
                    methods_csv_filename = f'{constants.RBO_QPP_RESULTS_FOLDER}/group_size_{group_size}_methods_comparison_{timestamp}.csv'
            methods_df.to_csv(methods_csv_filename, index=False)
            print(f"Method comparison results saved to {methods_csv_filename}")
            
            # Print table preview
            print("\nMethod Comparison Summary:")
            print("="*40)
            print(f"{'Method':<12} {'tau':<10} {'delta_tau':<10}")
            print("-"*32)
            for _, row in methods_df.iterrows():
                print(f"{row['Method']:<12} {row['tau']:<10.4f} {row['delta_tau']:<10.4f}")
            print("="*40)
    
    print('--------------------------------')
    return detailed_df, group_tau_results

def main():

    # Load data
    rbo_qpp_data_loader = data_loader.RBOQPPDataLoader(queries_file=constants.QUERIES_TREC_FAIR, 
                                                    qrels_file=constants.QRELS_TREC_FAIR, 
                                                    res_file=constants.RES_TREC_FAIR, 
                                                    stopwords_file=constants.STOPWORDS_FILE, 
                                                    # ranking_lists_file=constants.RANKING_LISTS_UoGTrMabWeSA,
                                                    # ranking_lists_file=constants.RANKING_LISTS_UoGTrMabSaWR,
                                                    # ranking_lists_file=constants.RANKING_LISTS_UoGTrMabSaNR,
                                                    ranking_lists_file=constants.RANKING_LISTS_UoGTrMabSAED
                                                    )
    
    # Load index
    index = pt.IndexFactory.of(constants.TREC_FAIR_INDEX_PATH)
    
    # Initialize NQCSpecificity
    nqc_specificity = NQCSpecificity(index)
    uef_specificity = UEFSpecificity(nqc_specificity)  # Pass NQCSpecificity instance
    rsd_specificity = RSDSpecificity(nqc_specificity)
    # Get data
    ranking_lists_df = rbo_qpp_data_loader.ranking_lists_df
    queries_df = rbo_qpp_data_loader.queries_df
    res_df = rbo_qpp_data_loader.res_df
    # test_name = 'UoGTrMabSaWR'
    # test_name = 'UoGTrMabSaNR'
    # test_name = 'UoGTrMabWeSA'
    test_name = 'UoGTrMabSAED'
    # file_name = 'merged_df_UoGTrMabWeSA.csv'
    file_name = f'merged_df_{test_name}.csv'
    # Calculate missing scores
    if not os.path.exists(file_name):
        complete_df = compute_missing_scores(index, ranking_lists_df, queries_df, res_df, file_name)
        print(f"Missing score calculation completed, file saved to {file_name}")
    else:
        print(f"File already exists, reading file directly {file_name}")
        complete_df = pd.read_csv(file_name)
    complete_df['qid'] = complete_df['qid'].astype(str)
    complete_df['docno'] = complete_df['docno'].astype(str)
    group_size_list = [4]
    for i, group_size in enumerate(group_size_list):
        print('--------------------------------')
        time_start = time.time()
        print(f"Starting calculation for group size {group_size}", f"{len(group_size_list)-i-1} groups remaining")
        detailed_df, tau_results = experiment_process(queries_df, complete_df, group_size, nqc_specificity,
                                                     uef_specificity=uef_specificity,
                                                     rsd_specificity=rsd_specificity,
                                                     #   uncertainty=True,
                                                       use_normalization=True,
                                                       normalization_method='sigmoid',
                                                       test_name=test_name
                                                       )
        time_end = time.time()
        print(f"Metric calculation time for group size {group_size}: {time_end - time_start:.2f} seconds")
        print('--------------------------------')


if __name__ == "__main__":
    main()
