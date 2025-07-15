import pandas as pd

from collections import Counter
from itertools import combinations

def filter_df_by_id(df, query_id):
    # Filter the DataFrame to only include rows with the specified query_id
    return df[df['id'] == query_id]

def get_ranklists_by_id(df, query_id, k_runs = 10):
    # For a given query_id, get the ranklists for each run (seq_no from 1 to k_runs)
    id_df = filter_df_by_id(df, query_id)
    ranklists = []
    for seq in range(1, k_runs + 1):
        seq_df = id_df[id_df['seq_no'] == seq]
        ranklists.append(seq_df['page_id'].tolist())
    return ranklists

def get_page_ids_with_min_count(df, query_id, min_count=2):
    # For a given query_id, return page_ids that appear at least min_count times across all ranklists
    id_df = filter_df_by_id(df, query_id)
    ranklists = get_ranklists_by_id(df, query_id)
    all_page_ids = []
    for ranklist in ranklists:
        all_page_ids.extend(ranklist)
    page_id_counts = Counter(all_page_ids)
    page_ids = [pid for pid, count in page_id_counts.items() if count >= min_count]
    return page_ids

def rbo(list1, list2, p=0.9, k=None):
    """
    Compute the Rank-Biased Overlap (RBO) similarity score, following Webber 2010.
    list1, list2: ordered lists
    p: persistence parameter
    k: truncation depth
    Returns a value in [0, 1].
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
    Compute the average local RBO score for a given page_id across multiple ranklists.
    Args:
        page_id: target page ID
        ranklists: list of ordered lists
        window: local window width
        p: RBO parameter
    Returns:
        Average RBO score, or None if the page_id appears in fewer than two ranklists
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
    for page_id in page_ids:
        sum_rbo += local_rbo_for_doc(page_id, ranklists)
    return sum_rbo / len(page_ids)

# print(average_local_rbo(page_ids, ranklists))

def main():
    # Main function to read the data, process each query_id, and print the results
    df = pd.read_csv('stochastic_runs/input.UoGTrMabSAED', sep='\t')
    unique_ids = [int(i) for i in df['id'].unique().tolist()]
    k_runs = 100
    for query_id in unique_ids:
        page_ids = get_page_ids_with_min_count(df, query_id)
        ranklists = get_ranklists_by_id(df, query_id, k_runs=k_runs)
        print(query_id, average_local_rbo(page_ids, ranklists))

main()