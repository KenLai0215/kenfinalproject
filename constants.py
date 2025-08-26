import os

# Base path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# new_queries_and_qrels data file paths
# QUERIES_DL19 = os.path.join(BASE_DIR, "orginal_queries and qrels", "pass_2019.queries")
# QRELS_DL19 = os.path.join(BASE_DIR, "orginal_queries and qrels", "pass_2019.qrels")
# QUERIES_DL20 = os.path.join(BASE_DIR, "orginal_queries and qrels", "pass_2020.queries")
# QRELS_DL20 = os.path.join(BASE_DIR, "orginal_queries and qrels", "pass_2020.qrels")

# old settings
QUERIES_DL1920 = os.path.join(BASE_DIR, "orginal_queries and qrels", "trecdl1920.queries")
QRELS_DL1920 = os.path.join(BASE_DIR, "orginal_queries and qrels", "trecdl1920.qrels")
QUERIES_TREC_FAIR = os.path.join(BASE_DIR, "fair_ir", "topics.tsv")
QRELS_TREC_FAIR = os.path.join(BASE_DIR, "fair_ir", "qrels.txt")
RES_TREC_FAIR = os.path.join(BASE_DIR, "fair_ir", "runs", "UoGRelvOnlyT1_lmjm.res")

# Random sampling result file paths
RANKING_LISTS_UoGTrMabWeSA = os.path.join(BASE_DIR, "fair_ir", "stochastic_runs", "input.UoGTrMabWeSA")
RANKING_LISTS_UoGTrMabSaWR = os.path.join(BASE_DIR, "fair_ir", "stochastic_runs", "input.UoGTrMabSaWR")
RANKING_LISTS_UoGTrMabSaNR = os.path.join(BASE_DIR, "fair_ir", "stochastic_runs", "input.UoGTrMabSaNR")
RANKING_LISTS_UoGTrMabSAED = os.path.join(BASE_DIR, "fair_ir", "stochastic_runs", "input.UoGTrMabSAED")

# rbo_qpp results save path
RBO_QPP_RESULTS_FOLDER = os.path.join(BASE_DIR, "rbo_qpp_results")
os.makedirs(RBO_QPP_RESULTS_FOLDER, exist_ok=True)

# Retrieval result file paths
# BM25_Top100_DL1920 = os.path.join(BASE_DIR, "trecdl1920.bm25.res")
# ColBERT_Top100_DL1920 = os.path.join(BASE_DIR, "trecdl1920.colbert-e2e.res")
trec_fair_res_folder = os.path.join(BASE_DIR, "fair_ir", "runs")  # TREC Fair results folder

# Index paths
# MSMARCO_INDEX_PATH = os.path.join(BASE_DIR, "pyterrier_msmarco_index")
TREC_FAIR_INDEX_PATH = os.path.join(BASE_DIR, "pyterrier_trec_fair_index")

# Original data paths
trec_fair_orignal_data = os.path.join(BASE_DIR, "orginal_data", "coll.jsonl")
# METADATA_FILE = os.path.join(BASE_DIR, "orginal_data", "metadata.jsonl")
STOPWORDS_FILE = os.path.join(BASE_DIR, "orginal_data", "stop.txt")
MSMARCO_Original_data = os.path.join(BASE_DIR, "orginal_data", "collection.tsv")
trec_fair_evals_folder = os.path.join(BASE_DIR, "fair_ir", "evals")
trec_fair_stochastic_runs_folder = os.path.join(BASE_DIR, "fair_ir", "stochastic_runs")

# Experiment parameters
CUTOFFS = [50]  # Cutoff value list
DEFAULT_RETRIEVAL_CUTOFF = 50  # Default retrieval cutoff - consider top-k search results
NUM_SAMPLES = 100  # Number of samples
WRITE_PERMS = False  # Whether to write permutation files

# Lambda value for Language Model with Jelinek-Mercer smoothing weight calculation in trec_fair
lambda_value = 0.2
trec_fair_res_folder = os.path.join(BASE_DIR, "fair_ir", "runs")

# 🔧 Random seed settings - ensure experiment reproducibility
RANDOM_SEED = 42  # Keep consistent with Java's Constants.SEED

# Constants corresponding to the Constants class in rank_swapper.py
NUM_SHUFFLES = 100  # Corresponds to Constants.NUM_SHUFFLES in Java
TOPDOC_ALWAYS_SWAPPED = False  # Corresponds to Constants.TOPDOC_ALWAYS_SWAPPED in Java
ALLOW_UNSORTED_TOPDOCS = False  # Corresponds to Constants.ALLOW_UNSORTED_TOPDOCS in Java

# Evaluation metric cutoff settings in evaluator.py
# These values control the cutoff positions for various metrics, independent of retrieval cutoff
COMPUTE_AP_AT_K = 100       # AP@100 - Calculate average precision for top 100 documents
COMPUTE_NDCG_AT_K = 10      # nDCG@10 - Calculate normalized discounted cumulative gain for top 10 documents  
COMPUTE_RR_AT_K = 100       # RR@100 - Reciprocal rank cutoff (usually consistent with AP)
# Output path settings
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "stochastic-qpp-results")

# Sampling modes
SAMPLING_MODES = {
    'U': 'random',      # Uniform sampling
    'R': 'rel',         # Relevance-aware sampling  
    'M': 'av'           # Attribute-value based sampling
}

# RBO method calculation
RBO_P = 0.9

# Evaluation metrics
# METRICS = ['AP', 'nDCG', 'RR']

# # QPP method configuration
# QPP_SCORE_FILE_PREFIX = os.path.join(OUTPUT_BASE_DIR, "qpp_scores")

# Ensure output directories exist
# os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
# for mode in SAMPLING_MODES.values():
#     os.makedirs(os.path.join(OUTPUT_BASE_DIR, f"results-{mode}"), exist_ok=True) 