"""
Base class for Query Performance Prediction methods
"""
import nltk
from nltk.stem import PorterStemmer
import pandas as pd
import pyterrier as pt
from abc import ABC, abstractmethod
from typing import Dict, List
import os
import constants


class BaseQPPMethod(ABC):
    
    def __init__(self, index=None):
        """
        Initialize QPP method
        
        Args:
            index: PyTerrier index object
        """
        self.index = index
        self.data_source = None
        
        if not pt.started():
            pt.init()
    
    @abstractmethod
    def compute_specificity(self, query, top_docs: pd.DataFrame, cutoff: int = None) -> float:
        """
        Compute query specificity score
        
        Args:
            query: Query object (MsMarcoQuery or dict)
            top_docs: Retrieved document list
            cutoff: Cutoff value
            
        Returns:
            QPP prediction score
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of QPP method"""
        pass
    
    def set_data_source(self, data_source: str):
        """Set data source path"""
        self.data_source = data_source
    
    def write_permutation_map(self, queries: List, top_docs_map: Dict[str, pd.DataFrame], sample_id: int):
        """
        Write permutation mapping to file
        
        Args:
            queries: Query list
            top_docs_map: Mapping from query ID to document list
            sample_id: Sample ID
        """
        if not constants.WRITE_PERMS:
            return
        
        output_file = f"{constants.QPP_SCORE_FILE_PREFIX}/{self.name()}.{sample_id}.tsv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("qid\tqpp_score\n")
            
            for query in queries:
                if hasattr(query, 'get_id'):
                    qid = query.get_id()
                else:
                    qid = query['qid']
                
                top_docs = top_docs_map.get(qid, pd.DataFrame())
                if not top_docs.empty:
                    # Pre-compute QPP score and write to file
                    qpp_score = self.compute_specificity(query, top_docs, constants.CUTOFFS[0])
                    f.write(f"{qid}\t{qpp_score:.6f}\n")
    def stem_term(self, term: str) -> str:
        """
        Stem a term
        
        Args:
            term: Term to stem
            
        Returns:
            Stemmed term
        """
        if self.index is None:
            return term
        
        try:
            # Use PyTerrier stemmer
            stemmer = self.index.getStemmer()
            return stemmer.stem(term)
        except:
            return term
    
    def get_term_frequencies(self, docno: str) -> Dict[str, int]:
        """
        Get term frequency information for a document
        
        Args:
            docno: Document ID
            
        Returns:
            Term frequency dictionary
        """
        if self.index is None:
            return {}
        
        try:
            # Use PyTerrier to get document term frequency information
            meta_index = self.index.getMetaIndex()
            doc_id = meta_index.getDocument("docno", docno)
            term_frequencies = {}
            for term, le in self.index.getLexicon():
                if term in doc_id:
                    term_frequencies[term] = le.getFrequency()
            return term_frequencies
            
            # This needs to be adapted based on specific index structure
            # This is a simplified version
            return {}
        except:
            return {}
    
    def get_collection_frequency(self, term: str) -> int:
        """
        Get term frequency in the collection
        
        Args:
            term: Term
            
        Returns:
            Collection frequency
        """
        if self.index is None:
            return 0
        
        try:
            term = self.stem_term(term)
            lexicon = self.index.getLexicon()
            lexicon_entry = lexicon.getLexiconEntry(term)
            if lexicon_entry is not None:
                return lexicon_entry.getFrequency()
        except:
            pass
        return 0
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get number of documents containing the term
        
        Args:
            term: Term
            
        Returns:
            Document frequency
        """
        if self.index is None:
            return 0
        
        try:
            term = self.stem_term(term)
            lexicon = self.index.getLexicon()
            lexicon_entry = lexicon.getLexiconEntry(term)
            if lexicon_entry is not None:
                return lexicon_entry.getDocumentFrequency()
        except:
            pass
        return 0
    
    def get_collection_size(self) -> int:
        """Get collection size (total number of documents)"""
        if self.index is None:
            return 0
        
        try:
            return self.index.getCollectionStatistics().getNumberOfDocuments()
        except:
            return 0 