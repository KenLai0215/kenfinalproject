# RBO-QPP: Rank-Biased Overlap Query Performance Prediction

This project implements a novel Query Performance Prediction (QPP) method based on Rank-Biased Overlap (RBO) for Information Retrieval (IR) systems. It includes implementations of several baseline QPP methods (NQC, UEF, RSD) and the proposed RBO-QPP approach.

## Overview

Query Performance Prediction (QPP) aims to estimate the quality of search results without relevance judgments. This project introduces RBO-QPP, which leverages ranking similarities among multiple retrieval lists to improve prediction accuracy.

### Key Features

- **Multiple QPP Methods**: Implementations of NQC (Normalized Query Commitment), UEF (Utility Estimation Framework), RSD (Retrieval Score Distribution), and the novel RBO-QPP
- **Index Building**: Support for building indexes from JSONL and TSV formats using PyTerrier
- **Comprehensive Evaluation**: Includes various IR metrics (nDCG, AP, RR) and correlation measures (Kendall's tau)
- **TREC Fair Dataset Support**: Designed to work with TREC Fair Ranking datasets
- **Stochastic Ranking Generation**: Tools for generating perturbed rankings for robustness analysis

## Project Structure

```
.
├── rbo_qpp.py                  # Main program for RBO-QPP experiments
├── rebuild_indexes.py          # Index building tool for JSONL/TSV data
├── constants.py                # Configuration and constants
├── data_loader.py              # Data loading utilities
├── evaluator.py                # IR metrics evaluation
├── rank_swapper.py             # Ranking perturbation generator
├── data_analysis.py            # Result analysis tools
├── qpp_methods/                # QPP method implementations
│   ├── base_qpp.py            # Base QPP class
│   ├── nqc_specificity.py     # NQC implementation
│   ├── uef_specificity.py     # UEF implementation
│   └── rsd_specificity.py     # RSD implementation
├── trec_fair/                  # TREC Fair specific utilities
│   └── tf_collection.py       # Document retrieval for TREC Fair
├── fair_ir/                    # TREC Fair dataset files
│   ├── topics.tsv             # Query topics
│   ├── qrels.txt              # Relevance judgments
│   ├── runs/                  # Original run files
│   ├── stochastic_runs/       # Stochastic ranking samples
│   └── evals/                 # Evaluation results
└── orginal_data/               # Original data files
    ├── coll.jsonl             # TREC Fair collection
    ├── stop.txt               # Stopword list
    └── collection.tsv         # Alternative collection format
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Java 11 or higher (required by PyTerrier)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (for stemming):
```python
import nltk
nltk.download('punkt')
```

4. Download TREC Fair 2022 dataset:
   - Download `coll.jsonl` from: [TREC Fair 2022 Collection](https://gla-my.sharepoint.com/:u:/g/personal/2732980l_student_gla_ac_uk/Eb1loW7FAXpOslsVYVROyPcBdE8rd_M-pu7CtKajNbfaKw?e=3LmhWS)
   - Place the file in the `orginal_data/` directory (or update the path in `constants.py`)

## Usage

### Step 1: Build Indexes

Before running QPP experiments, you need to build the search index:

```bash
python rebuild_indexes.py
```

This will create indexes in:
- `pyterrier_trec_fair_index/` - TREC Fair JSONL index
- `pyterrier_msmarco_index/` - MS MARCO TSV index (if configured)

### Step 2: Run RBO-QPP Experiments

Run the main RBO-QPP program:

```bash
python rbo_qpp.py
```

The program will:
1. Load queries and relevance judgments
2. Compute missing document scores if needed
3. Run QPP experiments with different group sizes
4. Calculate correlations (Kendall's tau) between QPP predictions and actual performance
5. Save results to the `rbo_qpp_results/` directory

### Configuration

Edit `constants.py` to modify:
- File paths
- Experiment parameters (group sizes, cutoffs)
- RBO persistence parameter (RBO_P)
- Evaluation metrics settings

### Key Parameters

- `group_size`: Number of ranking lists per group (default: 4)
- `RBO_P`: RBO persistence parameter (default: 0.9)
- `use_normalization`: Whether to apply score normalization (default: True)
- `normalization_method`: Method for normalization ('sigmoid', 'tanh', 'cv_replace', etc.)

## Methodology

### RBO-QPP Algorithm

1. **Group Formation**: Divide multiple ranking lists into groups
2. **Consensus Finding**: Identify the "consensus ranking" within each group using pairwise RBO similarities
3. **Weight Calculation**: Compute RBO scores between each list and the consensus as weights
4. **QPP Aggregation**: Calculate weighted average of base QPP scores (NQC/UEF/RSD)

### Evaluation

The system evaluates QPP methods by:
- Computing correlation (Kendall's tau) between predicted and actual performance
- Calculating delta tau to measure stability
- Generating detailed metrics for analysis

## Output Files

The program generates several output files in `rbo_qpp_results/`:

- `*_detailed_metrics_analysis_*.csv`: Detailed metrics for each query and group
- `*_tau_analysis_by_group_*.csv`: Tau correlations by group
- `*_methods_comparison_*.csv`: Comparison of different QPP methods


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Data Source

The TREC Fair 2022 dataset used in this project can be downloaded from:
- Collection file (`coll.jsonl`): [Download from OneDrive](https://gla-my.sharepoint.com/:u:/g/personal/2732980l_student_gla_ac_uk/Eb1loW7FAXpOslsVYVROyPcBdE8rd_M-pu7CtKajNbfaKw?e=3LmhWS)

This dataset is provided by TREC Fair Ranking Track 2022.

## Acknowledgments

- PyTerrier team for the IR toolkit
- TREC organizers for the Fair Ranking dataset
- Original authors of NQC, UEF, and RSD methods

