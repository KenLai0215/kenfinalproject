import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

def filter_df_by_id(df, query_id):
    # Filter the DataFrame to keep only rows with the specified query_id
    return df[df['id'] == query_id]

def get_ranklists_by_id(df, query_id, k_runs = 10):
    # For the specified query_id, get the ranklist for each run (seq_no from 1 to k_runs)
    id_df = filter_df_by_id(df, query_id)
    ranklists = []
    for seq in range(1, k_runs + 1):
        seq_df = id_df[id_df['seq_no'] == seq]
        ranklists.append(seq_df['page_id'].tolist())
    return ranklists

def get_page_ids(df, query_id):
    # For the specified query_id, return all unique page_ids that appear in the ranklists (ignore frequency)
    id_df = filter_df_by_id(df, query_id)
    ranklists = get_ranklists_by_id(df, query_id)
    all_page_ids = []
    for ranklist in ranklists:
        all_page_ids.extend(ranklist)
    page_ids = list(set(all_page_ids))
    return page_ids

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

# print(rbo(ranklists[0], ranklists[1]))

def local_rbo_for_doc(page_id, ranklists, window=5, p=0.9):
    """
    Compute the average local RBO score for the given page_id across multiple ranklists.
    Args:
        page_id: target page ID
        ranklists: list of ordered lists
        window: local window width
        p: RBO parameter
    Returns:
        Average RBO score, or None if page_id appears less than twice
    """
    # Find the ranklists and positions where page_id appears
    locs = [(idx, rnk.index(page_id))
            for idx, rnk in enumerate(ranklists) if page_id in rnk]
    if len(locs) < 2:
        return None  # Not enough occurrences to compare

    scores = []
    for (i, ri), (j, rj) in combinations(locs, 2):
        sub_i = ranklists[i][max(0, ri-window): ri+window+1]
        sub_j = ranklists[j][max(0, rj-window): rj+window+1]
        scores.append(rbo(sub_i, sub_j, p=p))
    return sum(scores) / len(scores)

def average_local_rbo(page_ids, ranklists):
    # Compute the average local RBO score for all page_ids
    sum_rbo = 0
    count = 0
    for page_id in page_ids:
        score = local_rbo_for_doc(page_id, ranklists)
        if score is not None:
            sum_rbo += score
            count += 1
    if count == 0:
        return None
    return sum_rbo / count

# print(average_local_rbo(page_ids, ranklists))

def get_variance_of_page_ids(df, query_id, k_runs=10):
    """
    For a given query_id, find all page_ids that appear more than once,
    and collect their ranks (list positions, starting from 1) in each seq_no.
    Returns the average, minimum, and maximum variance of ranks for all valid page_ids.
    """
    # Filter data for the specified query_id
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

def main():
    # Main function to read the data, process each query_id, and print the results
    df = pd.read_csv('stochastic_runs/input.UoGTrMabSAED', sep='\t')
    unique_ids = [int(i) for i in df['id'].unique().tolist()]
    k_runs = 10
    for query_id in unique_ids:
        page_ids = get_page_ids(df, query_id)
        ranklists = get_ranklists_by_id(df, query_id, k_runs=k_runs)
        print('query_id:', query_id, 'average_local_rbo:', average_local_rbo(page_ids, ranklists))
        average_variance, min_variance, max_variance = get_variance_of_page_ids(df, query_id, k_runs=k_runs)
        print('query_id:', query_id, 'average_variance:', average_variance, 'min_variance:', min_variance, 'max_variance:', max_variance)
main()