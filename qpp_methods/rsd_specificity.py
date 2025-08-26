"""
RSD Specificity QPP Method
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Set
from .base_qpp import BaseQPPMethod
from .nqc_specificity import NQCSpecificity


class RSDSpecificity(BaseQPPMethod):
    """
    RSD Specificity QPP method
    
    Computes ranking overlap (RBO) and weighted average QPP scores based on sampling
    """
    
    def __init__(self, base_qpp_method: NQCSpecificity):
        """
        Initialize RSD specificity method
        
        Args:
            base_qpp_method: Base QPP method instance
        """
        super().__init__(base_qpp_method.index)
        self.qpp_method = base_qpp_method
        
        # Constants
        self.NUM_SAMPLES = 10
        # Set random seed for reproducibility
        random.seed(42)
    
    def name(self) -> str:
        """Return method name"""
        return "RSD"
    
    def sample_top_docs(self, top_docs: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Sample TopDocs
        
        Args:
            top_docs: Original retrieval results
            k: Sample size
            
        Returns:
            Sampled retrieval results
        """
        if top_docs.empty:
            return top_docs
        
        # Ensure sampling ratio doesn't exceed 80% to avoid sampling all documents
        max_sample_ratio = 0.8
        max_sample_size = max(1, int(len(top_docs) * max_sample_ratio))
        sample_size = min(len(top_docs), k, max_sample_size)
        
        # Ensure at least some variation: if only 1 document, return directly
        if len(top_docs) <= 1:
            return top_docs.copy()
            
        # Shuffle documents
        shuffled_docs = top_docs.sample(frac=1.0, random_state=random.randint(0, 10000))
        
        # Take top k documents
        sampled_docs = shuffled_docs.head(sample_size).copy()
        
        # Reassign rankings
        sampled_docs = sampled_docs.reset_index(drop=True)
        sampled_docs['rank'] = range(1, len(sampled_docs) + 1)
        
        return sampled_docs
    
    def compute_rbo(self, top_docs1: pd.DataFrame, top_docs2: pd.DataFrame, p: float = 0.9) -> float:
        """
        Compute Rank-Biased Overlap (RBO)
        
        Args:
            top_docs1: First ranking list
            top_docs2: Second ranking list  
            p: RBO parameter, default 0.9
            
        Returns:
            RBO value
        """
        if top_docs1.empty or top_docs2.empty:
            return 0.0
        
        # Create ranking mappings
        rank_map1 = {row['docno']: row['rank'] for _, row in top_docs1.iterrows()}
        rank_map2 = {row['docno']: row['rank'] for _, row in top_docs2.iterrows()}
        
        # Get maximum length of both lists
        max_len = max(len(rank_map1), len(rank_map2))
        if max_len == 0:
            return 1.0
        
        # Calculate position-sensitive overlap
        overlap_score = 0.0
        for depth in range(1, max_len + 1):
            # Get document sets at current depth
            docs1_at_depth = {doc for doc, rank in rank_map1.items() if rank <= depth}
            docs2_at_depth = {doc for doc, rank in rank_map2.items() if rank <= depth}
            
            # Calculate overlap ratio
            if docs1_at_depth or docs2_at_depth:
                union_size = len(docs1_at_depth.union(docs2_at_depth))
                intersection_size = len(docs1_at_depth.intersection(docs2_at_depth))
                if union_size > 0:
                    overlap_at_depth = intersection_size / union_size
                else:
                    overlap_at_depth = 0.0
                
                # Apply position decay weight
                weight = (p ** (depth - 1)) * (1 - p)
                overlap_score += weight * overlap_at_depth
        
        return overlap_score
    
    def compute_specificity(self, query, top_docs: pd.DataFrame, cutoff: int = None) -> float:
        """
        Compute RSD specificity score
        
        Args:
            query: Query object or dictionary
            top_docs: Retrieved document list
            cutoff: Cutoff value
            
        Returns:
            RSD score
        """
        if top_docs.empty or (cutoff is not None and cutoff <= 0):
            return 0.0
        
        # Limit to cutoff number of documents, use all documents if cutoff is None
        if cutoff is None:
            docs_subset = top_docs
        else:
            docs_subset = top_docs.head(cutoff)
        
        if len(docs_subset) == 0:
            return 0.0
        
        avg_rank_sim = 0.0
        
        # Sample NUM_SAMPLES times
        for i in range(self.NUM_SAMPLES):
            # Use smaller sample size to ensure variability
            rlm_num_top_docs = min(15, len(docs_subset))
            sampled_top_docs = self.sample_top_docs(docs_subset, rlm_num_top_docs)
            
            # Compute QPP estimate
            effective_cutoff = len(docs_subset) if cutoff is None else cutoff
            qpp_estimate = self.qpp_method.compute_specificity(query, sampled_top_docs, effective_cutoff)
            
            # Compute ranking similarity
            rank_sim = self.compute_rbo(docs_subset, sampled_top_docs)
            
            # Weight by ranking similarity
            w = rank_sim * qpp_estimate
            avg_rank_sim += w
        
        # Return average weighted score
        return avg_rank_sim / self.NUM_SAMPLES 