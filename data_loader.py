"""
Data loader for queries, qrels and retrieval results
"""

import pandas as pd
import pyterrier as pt
from typing import Dict, List, Tuple
import constants
import math
import re
from collections import Counter
import nltk

class DataLoader:
    """Data loader for handling queries, qrels and retrieval results"""
    
    def __init__(self):
        if not pt.java.started():
            pt.init()
    
    def load_queries(self, query_file: str = None) -> pd.DataFrame:
        """Load query file"""
        if query_file is None:
            query_file = constants.QUERIES_DL1920
        
        queries = []
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    query = parts[1]
                    queries.append({'qid': str(qid), 'query': query})
        
        return pd.DataFrame(queries)
    
    def load_qrels(self, qrels_file: str = None) -> pd.DataFrame:
        """Load qrels file"""
        if qrels_file is None:
            qrels_file = constants.QRELS_DL1920
        
        qrels = []
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()  # Use space separator instead of tab
                if len(parts) >= 4:
                    qid = parts[0]
                    # parts[1] is usually "Q0", skip it
                    docid = parts[2]
                    label = int(parts[3])
                    qrels.append({'qid': str(qid), 'docno': docid, 'label': label})
        
        return pd.DataFrame(qrels)
    
    def load_retrieval_results(self, res_file: str) -> pd.DataFrame:
        """Load retrieval results file"""
        results = []
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = parts[0]
                    # parts[1] is usually "Q0", skip it
                    docid = parts[2]
                    rank = int(parts[3])
                    score = float(parts[4])
                    # parts[5] is system name, can be ignored
                    results.append({
                        'qid': str(qid), 
                        'docno': docid, 
                        'rank': rank, 
                        'score': score
                    })
        
        df = pd.DataFrame(results)
        # Keep original rank order, no re-sorting or rank reassignment
        return df
    
    def get_top_docs_by_query(self, results_df: pd.DataFrame, top_k= None) -> Dict[str, pd.DataFrame]:
        """Group retrieval results by query ID, take top_k documents for each query"""
        top_docs_map = {}
        if top_k is None:
            top_k = results_df.shape[0]
        for qid, group in results_df.groupby('qid'):
            # Take top_k results
            top_docs = group.head(top_k).copy()
            top_docs_map[qid] = top_docs
        return top_docs_map


class OneStepRetriever:
    
    def __init__(self, query_file: str, res_file: str, qrels_file= constants.QRELS_DL1920, index_path= constants.MSMARCO_INDEX_PATH):
        self.data_loader = DataLoader()
        self.queries_df = self.data_loader.load_queries(query_file)
        self.results_df = self.data_loader.load_retrieval_results(res_file)
        self.qrels_df = self.data_loader.load_qrels(qrels_file)
        self.top_docs_map = self.data_loader.get_top_docs_by_query(self.results_df)
        
        # Initialize index
        self.index = pt.IndexFactory.of(index_path)
        
    def get_query_list(self) -> List[Dict]:
        """Get query list"""
        return self.queries_df.to_dict('records')
    
    def get_index(self):
        """Get PyTerrier index"""
        return self.index
    
    def get_results_for_query(self, qid: str) -> pd.DataFrame:
        """Get retrieval results for specific query"""
        return self.top_docs_map.get(qid, pd.DataFrame())


class MsMarcoQuery:
    """Corresponds to MsMarcoQuery class in Java"""
    
    def __init__(self, qid: str, query_text: str):
        self.qid = qid
        self.query_text = query_text
    
    def get_id(self) -> str:
        return self.qid
    
    def get_query(self) -> str:
        return self.query_text
    
    @classmethod
    def from_dict(cls, query_dict: Dict):
        return cls(query_dict['qid'], query_dict['query']) 
    

class RBOQPPDataLoader:
    """RBO-QPP data loader class for loading and preprocessing RBO-related data"""
    
    def __init__(self, queries_file: str, qrels_file: str = None, res_file: str = None, stopwords_file: str = None, ranking_lists_file: str = None):
        """
        Initialize RBO-QPP data loader
        
        Args:
            queries_file: Query file path
            qrels_file: Qrels file path (optional)
            res_file: Retrieval results file path (optional)
        """
        self.data_loader = DataLoader()
        self.porter = nltk.PorterStemmer()
        
        # Load query data
        self.queries_df = self.load_queries(queries_file)
        
        # Load qrels and results files if provided
        if qrels_file:
            self.qrels_df = self.data_loader.load_qrels(qrels_file)
        else:
            self.qrels_df = None
            
        if res_file:
            self.res_df = self.data_loader.load_retrieval_results(res_file)
        else:
            self.res_df = None
        if ranking_lists_file:
            self.ranking_lists_df = self.load_ranking_lists(ranking_lists_file)
        else:
            self.ranking_lists_df = None

    def stopwords_remover(self, text: str) -> str:
        """
        Remove stopwords
        
        Args:
            text: Input text
            
        Returns:
            str: Text after stopwords removal
        """
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                stopwords = f.read().splitlines()
            text = re.sub(r'\b(?:{})\b'.format('|'.join(stopwords)), '', text, flags=re.IGNORECASE)
            return text
        except FileNotFoundError:
            # Return original text if stopwords file doesn't exist
            return text

    def load_queries(self, queries_file: str) -> pd.DataFrame:
        """
        Load query data and perform preprocessing
        
        Args:
            queries_file: Query file path
            
        Returns:
            pd.DataFrame: Preprocessed query data
        """
        queries = pd.read_csv(queries_file, header=None, sep='\t', names=['qid', 'query'])
        queries['qid'] = queries['qid'].astype(str)
        
        # Remove stopwords
        queries['query'] = queries['query'].apply(lambda x: self.stopwords_remover(x))
        
        # Stemming
        queries['query'] = queries['query'].apply(lambda x: self.porter.stem(x))
        
        
        return queries

    def load_ranking_lists(self, ranking_lists_file: str) -> pd.DataFrame:
        """
        Load ranking lists data
        
        Args:
            ranking_lists_file: Ranking lists file path
            
        Returns:
            pd.DataFrame: Ranking lists data
        """
        ranking_lists = pd.read_csv(ranking_lists_file, header=None, sep='\t', names=['qid', 'seq_no', 'docno'])
        # Skip first row (header row)
        ranking_lists = ranking_lists.iloc[1:]
        ranking_lists['qid'] = ranking_lists['qid'].astype(str)
        ranking_lists['docno'] = ranking_lists['docno'].astype(str)
        return ranking_lists

    def load_qrels(self, qrels_file: str) -> pd.DataFrame:
        qrels = pd.read_csv(qrels_file, header=None, sep='\t', names=['qid', 'iter', 'docno', 'relevance'])
        qrels['qid'] = qrels['qid'].astype(str)
        qrels['docno'] = qrels['docno'].astype(str)
        return qrels
    
    def load_res(self, res_file: str) -> pd.DataFrame:
        res = pd.read_csv(res_file, header=None, sep='\t', names=['qid', 'Q0', 'docno', 'rank', 'score', 'runid'])
        res['qid'] = res['qid'].astype(str)
        res['docno'] = res['docno'].astype(str)
        return res