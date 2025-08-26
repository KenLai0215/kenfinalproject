"""
NQC (Normalized Query Commitment) Specificity QPP Method
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Set
from .base_qpp import BaseQPPMethod


class NQCSpecificity(BaseQPPMethod):
    """NQC (Normalized Query Commitment) specificity method implementation"""
    
    def __init__(self, index=None, use_normalization=False):
        """
        Initialize NQC specificity method
        
        Args:
            index: PyTerrier index object
            use_normalization: Whether to use normalization
        """
        super().__init__(index)
        self.use_normalization = use_normalization
    
    def name(self) -> str:
        """Return method name"""
        return "nqc"
    
    def compute_specificity(self, query, top_docs: pd.DataFrame, cutoff: int = None) -> float:
        """
        Compute NQC specificity score
        
        Args:
            query: Query object or dictionary
            top_docs: Retrieved document list
            cutoff: Cutoff value
            
        Returns:
            NQC score
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
        
        # Get score list
        scores = docs_subset['score'].values
        
        # Calculate NQC (consistent with Java version)
        effective_cutoff = len(docs_subset) if cutoff is None else cutoff
        nqc_score = self.compute_nqc(query, scores, effective_cutoff)
        
        # Apply normalization if enabled
        if self.use_normalization:
            nqc_score = self.normalize_nqc(scores, nqc_score)
        
        return nqc_score
    
    def compute_nqc(self, query, rsvs: np.ndarray, k: int) -> float:
        """
        Compute NQC, consistent with Java version
        
        Corresponds to Java method: computeNQC(Query q, double[] rsvs, int k)
        
        Args:
            query: Query object
            rsvs: Document score array (corresponds to Java rsvs)
            k: Cutoff value
            
        Returns:
            NQC score
        """
        # Limit to k scores (corresponds to Java Arrays.stream(rsvs).limit(k).toArray())
        rsvs = rsvs[:k] if len(rsvs) > k else rsvs
        
        if len(rsvs) == 0:
            return 0.0
        
        # If only one score, variance is 0
        if len(rsvs) == 1:
            return 0.0
        
        # Calculate reference value (mean) - corresponds to Java: ref = Arrays.stream(rsvs).average().getAsDouble()
        ref = np.mean(rsvs)
        
        # Calculate variance - corresponds to Java nqc calculation part
        nqc = 0.0
        for rsv in rsvs:
            delta = rsv - ref  # corresponds to Java: del = rsv - ref
            nqc += delta * delta  # corresponds to Java: nqc += del*del
        nqc /= len(rsvs)  # corresponds to Java: nqc /= (double)rsvs.length
        
        # Calculate average IDF - corresponds to Java: avgIDF = Arrays.stream(idfs(q)).average().getAsDouble()
        avg_idf = self._get_average_idf_java_version(query)
        
        # Return variance * average IDF - corresponds to Java: return nqc * avgIDF
        final_nqc = nqc * avg_idf
        
        return final_nqc
    
    def _get_average_idf_java_version(self, query) -> float:
        """
        Get average IDF value for query, consistent with Java version idfs method
        
        Corresponds to Java: Arrays.stream(idfs(q)).average().getAsDouble()
        
        Args:
            query: Query object
            
        Returns:
            Average IDF value
        """
        if self.index is None:
            return 1.0  # corresponds to Java: reader!=null? ... : 1.0
        
        try:
            # Get query text
            query_text = self._extract_query_text(query)
            if not query_text:
                return 1.0
            
            # Get query terms (corresponds to Java extractTerms)
            query_terms = self._extract_query_terms(query_text)
            if not query_terms:
                return 1.0
            
            # Calculate IDF for each term (corresponds to Java idfs method)
            idfs = self._compute_idfs(query_terms)
            
            # Return average IDF (corresponds to Java Arrays.stream(idfs).average().getAsDouble())
            avg_idf = np.mean(idfs) if idfs else 1.0
            return avg_idf
            
        except Exception as e:
            # Corresponds to Java catch block, return default value 1.0
            return 1.0
    
    def _extract_query_text(self, query) -> str:
        """Extract query text"""
        # If dictionary format
        if isinstance(query, dict):
            # Priority: 'query' field, then 'text' field
            if 'query' in query:
                return str(query['query'])
            elif 'text' in query:
                return str(query['text'])
            elif 'query_text' in query:
                return str(query['query_text'])
            else:
                return ""
        
        # If object format
        if hasattr(query, 'get_query'):
            return str(query.get_query())
        elif hasattr(query, 'query'):
            return str(query.query)
        elif hasattr(query, 'text'):
            return str(query.text)
        elif hasattr(query, 'query_text'):
            return str(query.query_text)
        
        # If pandas Series (DataFrame row)
        if hasattr(query, 'get') and hasattr(query, 'index'):
            if 'query' in query:
                return str(query['query'])
            elif 'text' in query:
                return str(query['text'])
            elif 'query_text' in query:
                return str(query['query_text'])
        
        # Finally convert to string directly, but check if it looks like qid
        result = str(query).strip()
        
        # Check if it looks like qid (pure numbers)
        if result.isdigit():
            return ""
        
        return result
    
    def _extract_query_terms(self, query_text: str) -> Set[str]:
        """
        Extract query terms, simulating Java version extractTerms method
        
        Corresponds to Java: q.createWeight(searcher, ScoreMode.COMPLETE, 1).extractTerms(qterms)
        
        Args:
            query_text: Query text
            
        Returns:
            Query terms set
        """
        # Use PyTerrier index for tokenization to ensure consistency with index building
        if self.index is not None and hasattr(self.index, 'getTokeniser'):
            try:
                # Use index tokeniser for tokenization
                tokeniser = self.index.getTokeniser()
                # PyTerrier/Terrier tokeniser automatically handles:
                # - Lowercase conversion
                # - Stopword removal (if enabled during index building)
                # - Stemming (if enabled during index building)
                # - Number processing (according to index configuration)
                terms = []
                tokens = tokeniser.tokenise(query_text)
                while tokens.hasNext():
                    term = tokens.next()
                    if term:  # Filter empty terms
                        terms.append(term)
                return set(terms)
            except Exception as e:
                # If tokeniser call fails, fallback to simple tokenization
                pass
        
        # Fallback: simple tokenization
        # Note: This may not be consistent with actual index processing
        import re
        
        # Convert to lowercase
        text = query_text.lower().strip()
        
        # Basic tokenization: split by spaces and punctuation
        # This simulates common text processing
        terms = re.findall(r'\b\w+\b', text)
        
        # Remove pure numbers (according to index configuration)
        # From test results, index may filter numbers as their DF are all 0
        terms = [term for term in terms if not term.isdigit()]
        
        # Remove duplicate terms (corresponds to Java Set<Term>)
        unique_terms = set(terms)
        
        # Filter empty terms
        return {term for term in unique_terms if term.strip()}
    
    def _compute_idfs(self, query_terms: Set[str]) -> List[float]:
        """
        Compute IDF values for query terms, consistent with Java version idfs method
        
        Corresponds to Java idfs method:
        - N = reader.numDocs()
        - idf = Math.log(N/(double)n)
        - if (n==0) n = 1; // avoid 0 error!
        
        Args:
            query_terms: Query terms set
            
        Returns:
            IDF values list
        """
        idfs = []
        
        # Get collection size (corresponds to Java N = reader.numDocs())
        collection_size = self.get_collection_size()
        if collection_size == 0:
            return [1.0] * len(query_terms)
        
        for term in query_terms:
            # Get document frequency (corresponds to Java n = reader.docFreq(t))
            doc_freq = self.get_document_frequency(term)
            
            # Avoid division by zero (corresponds to Java if (n==0) n = 1)
            if doc_freq == 0:
                doc_freq = 1
            
            # Calculate IDF (corresponds to Java Math.log(N/(double)n))
            idf = math.log(collection_size / doc_freq)
            idfs.append(idf)
        
        return idfs
    
    def get_rsvs(self, top_docs: pd.DataFrame, k: int) -> np.ndarray:
        """
        Extract score array from TopDocs, corresponds to Java getRSVs method
        
        Corresponds to Java: Arrays.stream(topDocs.scoreDocs).limit(k).map(scoreDoc -> scoreDoc.score).mapToDouble(d -> d).toArray()
        
        Args:
            top_docs: Retrieval results
            k: Cutoff value
            
        Returns:
            Score array
        """
        if top_docs.empty:
            return np.array([])
        
        # Limit to k documents and extract scores
        scores = top_docs.head(k)['score'].values
        return scores
    
    def normalize_nqc(self, scores: np.ndarray, nqc_score: float, eps: float = 1e-8) -> float:
        """
        Normalize NQC score using CV method
        
        Args:
            scores: Document score array
            nqc_score: Original NQC score
            eps: Small value to prevent division by zero
            
        Returns:
            Normalized NQC score
        """
        if len(scores) <= 1:
            return nqc_score
        
        # Calculate coefficient of variation (CV) normalization
        mu = float(np.mean(scores))
        sd = float(np.std(scores, ddof=0))
        cv = sd / (abs(mu) + eps)
        normalized_factor = cv / (1.0 + cv)
        
        # Combine NQC score with normalization factor
        return nqc_score * normalized_factor
    
    def nqc_cv_no_train(self, scores: np.ndarray, eps: float = 1e-8) -> float:
        """
        Calculate normalized NQC score based on coefficient of variation
        
        Args:
            scores: Document score array
            eps: Small value to prevent division by zero
            
        Returns:
            Normalized NQC score
        """
        if len(scores) <= 1:
            return 0.0
            
        mu = float(np.mean(scores))
        sd = float(np.std(scores, ddof=0))
        cv = sd / (abs(mu) + eps)
        return cv / (1.0 + cv)
    

class CumulativeNQC(BaseQPPMethod):
    """Cumulative NQC method (extended implementation)"""
    
    def __init__(self, index=None):
        super().__init__(index)
        self.base_nqc = NQCSpecificity(index)
    
    def name(self) -> str:
        return "CumNQC"
    
    def compute_specificity(self, query, top_docs: pd.DataFrame, cutoff: int = None) -> float:
        """
        Compute cumulative NQC score
        
        This method calculates NQC scores at different cutoff values and takes the average
        """
        if top_docs.empty or (cutoff is not None and cutoff <= 0):
            return 0.0
        
        # Calculate NQC at different cutoff values
        if cutoff is None:
            # If cutoff is None, use preset cutoff values
            cutoff_values = [10, 20, 30, len(top_docs)]
        else:
            cutoff_values = [min(cutoff, 10), min(cutoff, 20), min(cutoff, 30), cutoff]
        cutoff_values = list(set([c for c in cutoff_values if c > 0]))  # Remove duplicates and filter
        
        nqc_scores = []
        for c in cutoff_values:
            if c <= len(top_docs):
                nqc = self.base_nqc.compute_specificity(query, top_docs, c)
                nqc_scores.append(nqc)
        
        # Return average NQC score
        return np.mean(nqc_scores) if nqc_scores else 0.0