"""
Rank swapper for generating perturbed rankings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
import random
import math
import hashlib
from evaluator import Evaluator
import constants


class RankSwapper:
    """Rank swapper for relevance-aware ranking perturbation"""
    
    def __init__(self, qid: str, evaluator: Evaluator, top_docs: pd.DataFrame):
        """
        Initialize rank swapper
        
        Args:
            qid: Query ID
            evaluator: Evaluator instance
            top_docs: Original ranking document list
        """
        self.qid = qid
        self.evaluator = evaluator
        self.top_docs = top_docs.copy()
        
        # Ensure docno column type consistency
        self.top_docs['docno'] = self.top_docs['docno'].astype(str)
        
        # Get relevance labels
        query_qrels = self.evaluator.qrels_df[self.evaluator.qrels_df['qid'] == qid]
        query_qrels = query_qrels.copy()
        query_qrels['docno'] = query_qrels['docno'].astype(str)
        
        merged = self.top_docs.merge(
            query_qrels[['docno', 'label']], 
            on='docno', 
            how='left'
        )
        merged['label'] = merged['label'].fillna(0)
        self.labels = merged['label'].values
        
        # Pre-generate permutations
        self.permuted_top_docs = self.sample_permutations(qid, evaluator, top_docs)
    
    @staticmethod
    def hash_ranking(ranking: pd.DataFrame) -> str:
        """
        Calculate hash value of ranking for duplicate detection
        
        Args:
            ranking: Document ranking DataFrame
            
        Returns:
            Hash value of ranking
        """
        docno_sequence = '|'.join(ranking['docno'].astype(str).tolist())
        return hashlib.md5(docno_sequence.encode()).hexdigest()
    
    @staticmethod
    def generate_unique_samples(top_docs: pd.DataFrame, 
                               num_samples: int = 100, 
                               sampling_mode: str = 'U',
                               max_attempts: int = 1000) -> List[pd.DataFrame]:
        """
        Generate unique perturbed samples in batch
        
        Args:
            top_docs: Original ranking
            num_samples: Number of samples to generate
            sampling_mode: Sampling mode ('U', 'R', 'M')
            max_attempts: Maximum number of attempts
            
        Returns:
            List of unique perturbed samples
        """
        unique_samples = []
        seen_hashes = set()
        attempts = 0
        
        # Add original ranking as first sample
        original_hash = RankSwapper.hash_ranking(top_docs)
        seen_hashes.add(original_hash)
        unique_samples.append(top_docs.copy())
        
        while len(unique_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate sample based on mode
            if sampling_mode == 'U':
                sample = RankSwapper.uniform_prior_sampling_single_swap(top_docs)
            else:
                sample = RankSwapper.shuffle(top_docs, 1)
            
            # Check for duplicates using hash
            sample_hash = RankSwapper.hash_ranking(sample)
            
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unique_samples.append(sample)
        
        return unique_samples
    
    @staticmethod
    def uniform_prior_sampling_single_swap(top_docs: pd.DataFrame) -> pd.DataFrame:
        """
        Correct UPS implementation: swap one pair of documents including their scores
        
        Args:
            top_docs: Original ranking
            
        Returns:
            Ranking after swapping one pair of documents
        """
        shuffled_docs = top_docs.copy()
        n = len(shuffled_docs)
        
        if n <= 1:
            return shuffled_docs
        
        # Perform only one swap - follows paper UPS definition
        pos1 = int(random.random() * n)
        pos2 = RankSwapper.select_random_not_equal(pos1, n)
        
        # Swap complete row data (document ID, score, etc.) not just document ID
        row1_data = shuffled_docs.iloc[pos1].copy()
        row2_data = shuffled_docs.iloc[pos2].copy()
        
        # Swap the two rows' data while keeping rank column as position index
        shuffled_docs.iloc[pos1] = row2_data
        shuffled_docs.iloc[pos2] = row1_data
        
        # Update rank column to reflect new positions (starting from 1)
        shuffled_docs.loc[shuffled_docs.index[pos1], 'rank'] = pos1 + 1
        shuffled_docs.loc[shuffled_docs.index[pos2], 'rank'] = pos2 + 1
        
        return shuffled_docs
    
    def collect_relevance_positions(self):
        """
        Collect relevant and non-relevant document positions
        
        Returns:
            (rel_positions, nrel_positions): Lists of relevant and non-relevant positions
        """
        rel_positions = []
        nrel_positions = []
        
        # Handle TOPDOC_ALWAYS_SWAPPED
        if hasattr(constants, 'TOPDOC_ALWAYS_SWAPPED') and constants.TOPDOC_ALWAYS_SWAPPED:
            rel_positions.append(0)
        
        # Find positions of relevant and non-relevant documents
        for i, label in enumerate(self.labels):
            if label > 0:  # Relevant document
                rel_positions.append(i)
            else:  # Non-relevant document
                nrel_positions.append(i)
        
        return rel_positions, nrel_positions
    
    def generate_ras_sample(self, top_docs: pd.DataFrame) -> pd.DataFrame:
        """
        Standard RAS implementation: sample only from eligible relevant-nonrelevant pairs
        
        Args:
            top_docs: Original ranking (Anchor list)
            
        Returns:
            RAS perturbed ranking
        """
        rel_positions, nrel_positions = self.collect_relevance_positions()
        
        if not rel_positions or not nrel_positions:
            return top_docs.copy()  # Cannot perform RAS swap
        
        # Pre-filter all eligible relevant-nonrelevant pairs
        eligible_pairs = []
        for rel_pos in rel_positions:
            for nrel_pos in nrel_positions:
                # Only when non-relevant document ranks better (smaller position) can relevant document "move up"
                if nrel_pos < rel_pos:
                    eligible_pairs.append((rel_pos, nrel_pos))
        
        if not eligible_pairs:
            # No eligible swap pairs, return original ranking
            return top_docs.copy()
        
        # Randomly select one eligible pair for swapping
        rel_pos, nrel_pos = random.choice(eligible_pairs)
        
        # Swap documents (fixed version: swap complete row data including scores)
        swapped_docs = top_docs.copy()
        
        # Swap complete row data not just document ID
        row_rel = swapped_docs.iloc[rel_pos].copy()
        row_nrel = swapped_docs.iloc[nrel_pos].copy()
        
        # Swap complete data of the two rows
        swapped_docs.iloc[rel_pos] = row_nrel
        swapped_docs.iloc[nrel_pos] = row_rel
        
        # Update rank column to reflect new positions (starting from 1)
        swapped_docs.loc[swapped_docs.index[rel_pos], 'rank'] = rel_pos + 1
        swapped_docs.loc[swapped_docs.index[nrel_pos], 'rank'] = nrel_pos + 1
        
        return swapped_docs
    
    def sample_permutations(self, qid: str, evaluator: Evaluator, top_docs: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Generate permutations: remove swap condition restrictions
        
        Returns:
            List of pre-generated permutations
        """
        rel_positions, nrel_positions = self.collect_relevance_positions()
        permuted_docs_list = []
        
        # Ensure identity permutation exists
        permuted_docs_list.append(top_docs.copy())
        
        # Generate all possible relevant-nonrelevant document pair swaps unconditionally
        for rel_pos in rel_positions:
            for nrel_pos in nrel_positions:
                swapped_docs = self.swap_ranks(top_docs, rel_pos, nrel_pos)
                permuted_docs_list.append(swapped_docs)
        
        return permuted_docs_list
    
    def swap_ranks(self, top_docs: pd.DataFrame, rel_rank: int, nrel_rank: int) -> pd.DataFrame:
        """
        Swap documents at two positions including their scores (fixed version)
        
        Args:
            top_docs: Original document ranking
            rel_rank: Relevant document position
            nrel_rank: Non-relevant document position
            
        Returns:
            Document ranking after swap (both documents and scores correctly swapped)
        """
        # Copy original data
        swapped_docs = top_docs.copy()
        
        # Swap complete row data not just document ID
        row_rel = swapped_docs.iloc[rel_rank].copy()
        row_nrel = swapped_docs.iloc[nrel_rank].copy()
        
        # Swap complete data of the two rows
        swapped_docs.iloc[rel_rank] = row_nrel
        swapped_docs.iloc[nrel_rank] = row_rel
        
        # Update rank column to reflect new positions (starting from 1)
        swapped_docs.loc[swapped_docs.index[rel_rank], 'rank'] = rel_rank + 1
        swapped_docs.loc[swapped_docs.index[nrel_rank], 'rank'] = nrel_rank + 1
        
        return swapped_docs
    
    def sample(self) -> pd.DataFrame:
        """
        RAS mode: use standard paper real-time sampling
        
        Returns:
            RAS perturbed ranking
        """
        return self.generate_ras_sample(self.top_docs)
    
    @staticmethod
    def select_random_not_equal(k: int, M: int) -> int:
        """
        Select a random integer in range [0, M-1] that is not equal to k
        
        Args:
            k: Value to avoid
            M: Upper bound (exclusive)
            
        Returns:
            Randomly selected value
        """
        if k == 0:
            return 1 + int(random.random() * (M - 1))  # Select [1, M-1]
        if k == M - 1:
            return int(random.random() * (M - 1))  # Select [0, M-2]
        
        # Select from [0, k) or [k+1, M)
        if random.random() <= 0.5:
            return int(random.random() * k)  # Select [0, k)
        else:
            return (k + 1) + int(random.random() * (M - k - 1))  # Select [k+1, M)
    
    @staticmethod
    def shuffle(top_docs: pd.DataFrame, num_shuffles: int = None) -> pd.DataFrame:
        """
        Random shuffle following shuffle logic
        
        Args:
            top_docs: Original ranking document list
            num_shuffles: Number of swaps (default uses Constants.NUM_SHUFFLES=50)
            
        Returns:
            Randomly shuffled document ranking
        """
        if num_shuffles is None:
            num_shuffles = constants.NUM_SHUFFLES
            
        shuffled_docs = top_docs.copy()
        n = len(shuffled_docs)
        
        if n <= 1:
            return shuffled_docs
        
        # Perform specified number of random swaps
        for _ in range(num_shuffles):
            # Select first position
            rel_rank = 0 if constants.TOPDOC_ALWAYS_SWAPPED else int(random.random() * n)
            
            # Select second position different from first
            nrel_rank = RankSwapper.select_random_not_equal(rel_rank, n)
            
            # Swap document IDs, keep scores unchanged
            temp_docno = shuffled_docs.iloc[nrel_rank]['docno']
            shuffled_docs.iloc[nrel_rank, shuffled_docs.columns.get_loc('docno')] = shuffled_docs.iloc[rel_rank]['docno']
            
            if not constants.ALLOW_UNSORTED_TOPDOCS:
                shuffled_docs.iloc[rel_rank, shuffled_docs.columns.get_loc('docno')] = temp_docno
            else:
                # Complete swap
                shuffled_docs.iloc[rel_rank, shuffled_docs.columns.get_loc('docno')] = temp_docno
        
        return shuffled_docs


class Metadata:
    """Metadata class for handling document metadata"""
    
    def __init__(self, metadata_file: str = None):
        """
        Initialize metadata
        
        Args:
            metadata_file: Metadata file path
        """
        self.gender_value_map = {}
        
        if metadata_file:
            self.load_metadata(metadata_file)
    
    def load_metadata(self, metadata_file: str):
        """
        Load metadata file
        
        Args:
            metadata_file: Metadata file path
        """
        import json
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if count >= 1000:
                        break
                    
                    try:
                        json_data = json.loads(line.strip())
                        doc_id = str(json_data.get('page_id', ''))
                        gender = str(json_data.get('gender', ''))
                        
                        if gender and len(gender) > 0:
                            # Simplified gender determination logic
                            male = 'm' in gender.lower()
                            self.gender_value_map[doc_id] = male
                        
                        count += 1
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
    
    def is_male(self, doc_id: str) -> bool:
        """
        Determine if document corresponds to male
        
        Args:
            doc_id: Document ID
            
        Returns:
            Whether it is male
        """
        return self.gender_value_map.get(doc_id, False)


class AttributeValueBasedSwapper(RankSwapper):
    """Attribute value based ranking perturbation"""
    
    def __init__(self, qid: str, evaluator: Evaluator, top_docs: pd.DataFrame, metadata: Metadata):
        """
        Initialize attribute value based rank swapper
        
        Args:
            qid: Query ID
            evaluator: Evaluator instance
            top_docs: Original ranking document list
            metadata: Document metadata
        """
        self.metadata = metadata
        # Call parent constructor, but will use overridden sample_permutations method
        super().__init__(qid, evaluator, top_docs)
    
    def sample_permutations(self, qid: str, evaluator: Evaluator, top_docs: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Generate permutations following AttributeValueBasedSwapper logic
        
        Returns:
            List of pre-generated permutations
        """
        rel_ranks = []  # Use List not Set
        nrel_ranks = []
        permuted_docs_list = []
        
        # Ensure identity permutation exists
        permuted_docs_list.append(top_docs.copy())
        
        # Group based on metadata
        for i, (idx, row) in enumerate(top_docs.iterrows()):
            doc_id = row['docno']
            
            # Skip documents without metadata
            if doc_id not in self.metadata.gender_value_map:
                continue
                
            if self.metadata.is_male(doc_id):
                rel_ranks.append(i)
            else:
                nrel_ranks.append(i)
        
        # Generate permutations (condition: rel_rank > nrel_rank)
        for rel_rank in rel_ranks:
            for nrel_rank in nrel_ranks:
                if rel_rank > nrel_rank:
                    swapped_docs = self.swap_ranks(top_docs, rel_rank, nrel_rank)
                    permuted_docs_list.append(swapped_docs)
        
        return permuted_docs_list


class StochasticRankingGenerator:
    """Stochastic ranking generator integrating different perturbation strategies"""
    
    def __init__(self, sampling_mode: str = 'U', metadata: Metadata = None):
        """
        Initialize stochastic ranking generator
        
        Args:
            sampling_mode: Sampling mode ('U'=uniform, 'R'=relevance-aware, 'M'=metadata-based)
            metadata: Metadata object (for mode M)
        """
        self.sampling_mode = sampling_mode
        self.metadata = metadata
        self.swappers = {}
        self.cached_samples = {}  # Cache pre-generated samples
    
    def initialize_swappers(self, qid: str, evaluator: Evaluator, top_docs: pd.DataFrame):
        """Initialize rank swappers for specified query"""
        if self.sampling_mode == 'R':
            self.swappers[qid] = RankSwapper(qid, evaluator, top_docs)
        elif self.sampling_mode == 'M' and self.metadata:
            self.swappers[qid] = AttributeValueBasedSwapper(qid, evaluator, top_docs, self.metadata)
        # 'U' mode doesn't need specific swapper initialization
    
    def generate_batch_unique_samples(self, qid: str, top_docs: pd.DataFrame, 
                                    num_samples: int = 100, 
                                    evaluator: Evaluator = None) -> List[pd.DataFrame]:
        """
        Generate unique perturbed samples in batch
        
        Args:
            qid: Query ID
            top_docs: Original ranking document list
            num_samples: Number of samples to generate
            evaluator: Evaluator (required for R and M modes)
            
        Returns:
            List of unique perturbed samples
        """
        # Check if already cached
        cache_key = f"{qid}_{self.sampling_mode}_{num_samples}"
        
        if cache_key in self.cached_samples:
            return self.cached_samples[cache_key]
        
        if self.sampling_mode == 'U':
            # UPS mode: use corrected single swap implementation
            unique_samples = RankSwapper.generate_unique_samples(
                top_docs=top_docs,
                num_samples=num_samples,
                sampling_mode='U'
            )
        
        elif self.sampling_mode == 'R':
            # RAS mode: restore to standard paper real-time sampling implementation
            if evaluator is None:
                raise ValueError("RAS mode requires evaluator")
            
            if qid not in self.swappers:
                self.initialize_swappers(qid, evaluator, top_docs)
            
            unique_samples = []
            seen_hashes = set()
            max_attempts = num_samples * 20  # Increase attempt count
            attempts = 0
            
            # Add original ranking
            original_hash = RankSwapper.hash_ranking(top_docs)
            seen_hashes.add(original_hash)
            unique_samples.append(top_docs.copy())
            
            # Use correct RAS real-time sampling
            swapper = self.swappers[qid]
            
            while len(unique_samples) < num_samples and attempts < max_attempts:
                attempts += 1
                # Use standard paper RAS sampling: only swap eligible relevant-nonrelevant pairs
                sample = swapper.generate_ras_sample(top_docs)
                sample_hash = RankSwapper.hash_ranking(sample)
                
                if sample_hash not in seen_hashes:
                    seen_hashes.add(sample_hash)
                    unique_samples.append(sample)
        
        elif self.sampling_mode == 'M':
            # MAS mode: requires metadata and evaluator
            if evaluator is None or self.metadata is None:
                raise ValueError("MAS mode requires evaluator and metadata")
            
            if qid not in self.swappers:
                self.initialize_swappers(qid, evaluator, top_docs)
            
            unique_samples = []
            seen_hashes = set()
            max_attempts = num_samples * 10
            attempts = 0
            
            # Add original ranking
            original_hash = RankSwapper.hash_ranking(top_docs)
            seen_hashes.add(original_hash)
            unique_samples.append(top_docs.copy())
            
            while len(unique_samples) < num_samples and attempts < max_attempts:
                attempts += 1
                sample = self.swappers[qid].sample()
                sample_hash = RankSwapper.hash_ranking(sample)
                
                if sample_hash not in seen_hashes:
                    seen_hashes.add(sample_hash)
                    unique_samples.append(sample)
        
        else:
            raise ValueError(f"Unsupported sampling mode: {self.sampling_mode}")
        
        # Cache results
        self.cached_samples[cache_key] = unique_samples
        
        return unique_samples
    
    def generate_sample(self, qid: str, top_docs: pd.DataFrame, evaluator: Evaluator = None) -> pd.DataFrame:
        """
        Generate single perturbed sample for specified query
        
        Args:
            qid: Query ID
            top_docs: Original ranking document list
            evaluator: Evaluator (required for R mode)
            
        Returns:
            Perturbed document ranking
        """
        if self.sampling_mode == 'U':
            # Use standard paper UPS implementation (single swap)
            return RankSwapper.uniform_prior_sampling_single_swap(top_docs)
        elif self.sampling_mode == 'R':
            # R mode restore to standard paper real-time sampling
            if qid not in self.swappers and evaluator:
                self.initialize_swappers(qid, evaluator, top_docs)
            
            if qid in self.swappers:
                # Use real-time RAS sampling instead of pre-generated selection
                return self.swappers[qid].generate_ras_sample(top_docs)
            else:
                return RankSwapper.uniform_prior_sampling_single_swap(top_docs)
        elif self.sampling_mode == 'M' and qid in self.swappers:
            return self.swappers[qid].sample()
        else:
            # Default fallback to single swap
            return RankSwapper.uniform_prior_sampling_single_swap(top_docs)
    
    def get_sample_statistics(self, qid: str) -> Dict:
        """
        Get sample statistics
        
        Args:
            qid: Query ID
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'sampling_mode': self.sampling_mode,
            'cached_queries': list(self.cached_samples.keys()),
            'swapper_initialized': qid in self.swappers
        }
        
        # If cached samples exist, calculate statistics
        cache_key_prefix = f"{qid}_{self.sampling_mode}"
        for key in self.cached_samples:
            if key.startswith(cache_key_prefix):
                samples = self.cached_samples[key]
                stats['num_cached_samples'] = len(samples)
                stats['cache_key'] = key
                break
        
        return stats
    
    def clear_cache(self):
        """Clear all cache"""
        self.cached_samples.clear()
        self.swappers.clear()
    
    def clear_swappers(self):
        """Clear swapper cache"""
        self.swappers.clear()