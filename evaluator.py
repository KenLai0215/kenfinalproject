"""
Evaluator for computing retrieval effectiveness metrics
"""

import pandas as pd
import pyterrier as pt
from typing import Dict, List, Union
import numpy as np
from scipy.stats import kendalltau
import constants
from data_loader import DataLoader


class Metric:
    """Evaluation metrics enumeration"""
    AP = "AP"
    nDCG = "nDCG"
    RR = "RR"


class RankScore:
    """Rank score class for SARE calculation"""
    
    def __init__(self, id: int, rank: int, score: float):
        self.id = id
        self.rank = rank
        self.score = score
    
    def __lt__(self, other):
        # Sort by score in descending order
        return self.score > other.score


class NDCGCorrelation:
    """NDCG correlation calculation"""
    
    @staticmethod
    def exp_scaling(rel: float) -> float:
        return 2**rel - 1
    
    @staticmethod
    def linear_scaling(rel: float) -> float:
        return rel
    
    @staticmethod
    def compute_ndcg(labels: np.ndarray, scores: np.ndarray, scaling_function=None) -> float:
        if scaling_function is None:
            scaling_function = NDCGCorrelation.exp_scaling
            
        if len(labels) != len(scores) or len(labels) == 0:
            raise ValueError("Labels and scores must have the same length and be non-empty.")
        
        n = len(labels)
        indices = list(range(n))
        
        # Sort indices based on scores in descending order
        indices.sort(key=lambda i: -scores[i])
        
        dcg = NDCGCorrelation._compute_dcg(labels, indices, scaling_function)
        
        # Sort indices based on actual labels in descending order for ideal DCG
        indices.sort(key=lambda i: -labels[i])
        
        idcg = NDCGCorrelation._compute_dcg(labels, indices, scaling_function)
        
        return 0 if idcg == 0 else dcg / idcg
    
    @staticmethod
    def _compute_dcg(labels: np.ndarray, indices: List[int], scaling_function) -> float:
        dcg = 0.0
        for i, idx in enumerate(indices):
            dcg += scaling_function(labels[idx]) / np.log2(i + 2)
        return dcg
    
    def correlation(self, gt: np.ndarray, pred: np.ndarray) -> float:
        return self.compute_ndcg(gt, pred, self.exp_scaling)


class SARE:
    """SARE (Similarity-based Average Rank Error) calculation"""
    
    def compute_sare_per_query(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Calculate SARE value for each query based on rank differences"""
        n = len(gt)
        rank_diffs = np.zeros(n)
        
        # Create RankScore objects
        gt_rs = [RankScore(i, i, gt[i]) for i in range(n)]
        pred_rs = [RankScore(i, i, pred[i]) for i in range(n)]
        
        # Sort and assign ranks
        gt_rs.sort()
        pred_rs.sort()
        
        # Create mappings
        map_gts = {}
        map_preds = {}
        
        for i in range(n):
            gt_rs[i].rank = i
            pred_rs[i].rank = i
            map_gts[gt_rs[i].id] = gt_rs[i]
            map_preds[pred_rs[i].id] = pred_rs[i]
        
        # Calculate rank differences
        for id in map_gts.keys():
            gt_rank = map_gts[id].rank
            pred_rank = map_preds[id].rank
            rank_diffs[id] = abs(gt_rank - pred_rank) / len(gt)  # Normalized rank difference
            
        return rank_diffs
    
    def correlation(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Calculate average SARE value"""
        sare_per_query = self.compute_sare_per_query(gt, pred)
        return np.mean(sare_per_query)


class Evaluator:
    """Evaluator for computing retrieval effectiveness metrics"""
    
    def __init__(self, qrels_file: str, results_data: Union[str, Dict[str, pd.DataFrame]]):
        """
        Initialize evaluator
        
        Args:
            qrels_file: Path to qrels file
            results_data: Path to retrieval results file or results data dictionary
        """
        self.data_loader = DataLoader()
        self.qrels_df = self.data_loader.load_qrels(qrels_file)
        self.merage_df = pd.DataFrame()
        if isinstance(results_data, str):
            # Load from file
            self.results_df = self.data_loader.load_retrieval_results(results_data)
            self.top_docs_map = self.data_loader.get_top_docs_by_query(self.results_df)
        else:
            # Use provided data dictionary directly
            self.top_docs_map = results_data
            # Reconstruct as single DataFrame
            all_results = []
            for qid, docs in results_data.items():
                docs_copy = docs.copy()
                docs_copy['qid'] = qid
                all_results.append(docs_copy)
            self.results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def compute(self, qid: str, metric: str, cutoff: int = None, ap_cutoff: int = None, ndcg_cutoff: int = None, rr_cutoff: int = None) -> float:
        """
        Calculate specified metric value for specified query
        
        Important note:
        - qrels only contains annotations for relevant documents
        - Documents not appearing in qrels are considered non-relevant (label=0)
        - This is standard practice in information retrieval evaluation
        
        Args:
            qid: Query ID
            metric: Evaluation metric (AP, nDCG, RR)
            cutoff: Retrieval results cutoff, defaults to constants.DEFAULT_RETRIEVAL_CUTOFF
            ap_cutoff: AP calculation cutoff, defaults to constants.COMPUTE_AP_AT_K
            ndcg_cutoff: nDCG calculation cutoff, defaults to constants.COMPUTE_NDCG_AT_K
            rr_cutoff: RR calculation cutoff, defaults to constants.COMPUTE_RR_AT_K
            
        Returns:
            Metric value
        """
        # Use default values from constants
        if cutoff is None:
            cutoff = constants.DEFAULT_RETRIEVAL_CUTOFF
        if ap_cutoff is None:
            ap_cutoff = constants.COMPUTE_AP_AT_K
        if ndcg_cutoff is None:
            ndcg_cutoff = constants.COMPUTE_NDCG_AT_K
        if rr_cutoff is None:
            rr_cutoff = constants.COMPUTE_RR_AT_K
            
        # Ensure evaluation metric cutoff does not exceed retrieval cutoff
        # If evaluation cutoff is larger than retrieval cutoff, automatically use retrieval cutoff
        ap_cutoff = min(ap_cutoff, cutoff)
        ndcg_cutoff = min(ndcg_cutoff, cutoff)
        rr_cutoff = min(rr_cutoff, cutoff)
            
        # Get retrieval results
        query_results = self.top_docs_map.get(qid, pd.DataFrame())
        if query_results.empty:
            return 0.0
        
        # Limit to cutoff
        if cutoff is not None:
            query_results = query_results.head(cutoff)
        
        # Fix data type mismatch: unify docno as string
        query_results = query_results.copy()
        query_results['qid'] = query_results['qid'].astype(str)
        query_results['docno'] = query_results['docno'].astype(str)
        self.qrels_df['qid'] = self.qrels_df['qid'].astype(str)
        
        # Get relevance judgments (if no qrels record, all documents are marked as non-relevant)
        query_qrels = self.qrels_df[self.qrels_df['qid'] == qid]
        if query_qrels.empty:
            # If no qrels record for this query, all retrieval results are marked as non-relevant
            query_results['label'] = 0
            merged = query_results
        else:
            # Unify qrels docno type
            query_qrels_copy = query_qrels.copy()
            query_qrels_copy['docno'] = query_qrels_copy['docno'].astype(str)
            
            # Merge relevance labels (left join ensures all retrieval results are retained)
            merged = query_results.merge(
                query_qrels_copy[['docno', 'label']], 
                on='docno', 
                how='left'
            )
            # Key: Documents not appearing in qrels default to non-relevant (label=0)
            merged['label'] = merged['label'].fillna(0)
        
        if metric == Metric.AP:
            return self._compute_ap(merged['label'].values, ap_cutoff)
        elif metric == Metric.nDCG:
            return self._compute_ndcg(merged['label'].values, ndcg_cutoff)
        elif metric == Metric.RR:
            return self._compute_rr(merged['label'].values, rr_cutoff)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _compute_ap(self, labels: np.ndarray, ap_cutoff: int = 100) -> float:
        
        """
        Calculate Average Precision@k
        
        Args:
            labels: Relevance label array
            ap_cutoff: AP calculation cutoff, default is 100 (AP@100)
            
        Returns:
            AP@k value
        """
        if len(labels) == 0:
            return 0.0
        
        # Only consider first ap_cutoff documents
        labels_truncated = labels[:ap_cutoff]
        
        # Find positions of all relevant documents
        relevant_positions = np.where(labels_truncated > 0)[0]
        if len(relevant_positions) == 0:
            return 0.0
        
        # Calculate precision at each relevant position
        precisions = []
        for pos in relevant_positions:
            # precision@(pos+1) = number of relevant documents / (pos+1)
            relevant_so_far = np.sum(labels_truncated[:pos+1] > 0)
            precision = relevant_so_far / (pos + 1)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def _compute_ndcg(self, labels: np.ndarray, ndcg_cutoff: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Args:
            labels: Relevance label array
            ndcg_cutoff: nDCG calculation cutoff, default is 10 (nDCG@10)
            
        Returns:
            nDCG value
        """
        if len(labels) == 0:
            return 0.0
        
        # Calculate DCG@k
        dcg = 0.0
        for i, label in enumerate(labels[:ndcg_cutoff]):
            if label > 0:
                dcg += (2**label - 1) / np.log2(i + 2)
        
        # Calculate IDCG@k - using same cutoff
        sorted_labels = sorted(labels, reverse=True)
        idcg = 0.0
        for i, label in enumerate(sorted_labels[:ndcg_cutoff]):
            if label > 0:
                idcg += (2**label - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _compute_rr(self, labels: np.ndarray, rr_cutoff: int = 100) -> float:
        """
        Calculate Reciprocal Rank@k
        
        Args:
            labels: Relevance label array
            rr_cutoff: RR calculation cutoff, default is 100 (RR@100)
            
        Returns:
            RR@k value - reciprocal rank of first relevant document, returns 0 if no relevant document in first k documents
        """
        # Only consider first rr_cutoff documents
        labels_truncated = labels[:rr_cutoff]
        
        for i, label in enumerate(labels_truncated):
            if label > 0:
                return 1.0 / (i + 1)
        return 0.0
    def _return_df(self, qid: str) -> pd.DataFrame:
        return self.merage_df
    
    def get_all_retrieved_results(self) -> 'AllRetrievedResults':
        """Get all retrieval results, return wrapper class"""
        return AllRetrievedResults(self.top_docs_map)


class AllRetrievedResults:
    """All retrieved results wrapper class"""
    
    def __init__(self, top_docs_map: Dict[str, pd.DataFrame]):
        self.top_docs_map = top_docs_map
    
    def cast_to_top_docs(self) -> Dict[str, pd.DataFrame]:
        """Convert to TopDocs format (directly return pandas DataFrame dictionary here)"""
        return self.top_docs_map


class QPPMetricBundle:
    """QPP metric bundle for storing QPP evaluation results"""
    
    def __init__(self, target_metrics: np.ndarray = None, qpp_estimates: np.ndarray = None, 
                 tau: float = 0.0, ndcg: float = 0.0, per_query_sare: np.ndarray = None):
        """
        Initialize QPP metric bundle
        
        Args:
            target_metrics: Target metric value array
            qpp_estimates: QPP prediction value array  
            tau: Kendall's tau correlation
            ndcg: nDCG correlation
            per_query_sare: SARE value for each query
        """
        if target_metrics is not None and qpp_estimates is not None:
            self.target_metrics = target_metrics
            self.qpp_estimates = qpp_estimates
            
            # Calculate Kendall's tau correlation
            if len(target_metrics) > 1 and len(qpp_estimates) > 1:
                self.tau, _ = kendalltau(target_metrics, qpp_estimates)
                if np.isnan(self.tau):
                    self.tau = 0.0
            else:
                self.tau = 0.0
            
            # Calculate nDCG correlation
            ndcg_corr = NDCGCorrelation()
            self.ndcg = ndcg_corr.correlation(target_metrics, qpp_estimates)
            if np.isnan(self.ndcg):
                self.ndcg = 0.0
            
            # Calculate SARE for each query (based on rank differences)
            sare_calc = SARE()
            self.per_query_sare = sare_calc.compute_sare_per_query(target_metrics, qpp_estimates)
        else:
            self.tau = tau
            self.ndcg = ndcg
            self.per_query_sare = per_query_sare if per_query_sare is not None else np.array([])
    
    def sare(self) -> float:
        """Calculate average SARE value"""
        return np.mean(self.per_query_sare) if len(self.per_query_sare) > 0 else 0.0
    
    def sarc(self) -> float:
        """Calculate SARC value (1-SARE)"""
        return 1 - self.sare()
    
    def get_per_query_sare(self) -> np.ndarray:
        """Get SARE value for each query"""
        return self.per_query_sare
    
    def __str__(self):
        return f"QPPMetricBundle(tau={self.tau:.4f}, ndcg={self.ndcg:.4f}, n_queries={len(self.per_query_sare)})" 