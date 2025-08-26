import math
from nltk.stem import PorterStemmer
import constants
import re
import os
import pandas as pd
import time

class OptimizedDocumentRetriever:
    """Optimized document retriever - reduces redundant operations"""
    
    def __init__(self, index):
        self.direct_index = index.getDirectIndex()
        self.document_index = index.getDocumentIndex()
        self.lexicon = index.getLexicon()
        self.meta_index = index.getMetaIndex()
        self.collection_N = index.getCollectionStatistics().getNumberOfDocuments()
        self.stemmer = PorterStemmer()
        self.stopwords = self.get_stopwords()
        if self.direct_index is None:
            raise ValueError("Index does not have Direct Index")
        
        self.lexicon_cache = {}
    
    def num_data_clean(self, term):
        return re.sub(r'(-)?[0-9]+(\.[0-9]*)?', ' _NUM_ ', term)
    
    def get_stopwords(self):
        with open(constants.STOPWORDS_FILE, "rt", encoding="utf8") as f:
            return f.read().splitlines()
    
    def get_query_terms(self, query_text):
        query_terms = []
        for term in query_text.split():
            term = self.num_data_clean(term)
            stemmed_term = self.stemmer.stem(term)
            if stemmed_term in self.stopwords:
                continue
            query_terms.append(stemmed_term)
        return query_terms

    def get_document_terms(self, docno, query_terms):
        """Get query terms TF in document and complete document length"""
        
        internal_docid = self.meta_index.getDocument("docno", str(docno))
        doc_entry = self.document_index.getDocumentEntry(internal_docid)
        postings = self.direct_index.getPostings(doc_entry)
        
        query_terms_tf = {}
        doc_length = 0
        
        for posting in postings:
            term_id = posting.getId()
            frequency = posting.getFrequency()
            
            if term_id not in self.lexicon_cache:
                lexicon_entry = self.lexicon.getLexiconEntry(term_id)
                if lexicon_entry:
                    self.lexicon_cache[term_id] = lexicon_entry.getKey()
                else:
                    self.lexicon_cache[term_id] = None
            
            term_text = self.lexicon_cache[term_id]
            if term_text:
                doc_length += frequency
                if term_text in query_terms:
                    query_terms_tf[term_text] = frequency
        
        return query_terms_tf, doc_length

    def get_collection_frequency(self, terms):
        """Get collection frequency for terms"""
        
        terms_cf = {}
        try:
            for term in terms:
                lexicon_entry = self.lexicon.getLexiconEntry(term)
                if lexicon_entry is not None:
                    terms_cf[term] = lexicon_entry.getFrequency()
                else:
                    terms_cf[term] = 0
        except Exception as e:
            print(f"Error in get_collection_frequency for term '{term}': {e}")
            terms_cf[term] = 0
        return terms_cf

    def lmjm_score(self, docno, query_text, lambda_value = constants.lambda_value):
        query_terms = self.get_query_terms(query_text)
        query_terms_tf, single_doc_length = self.get_document_terms(docno, query_terms)
        
        existing_terms = [term for term in query_terms if query_terms_tf.get(term, 0) > 0]
        if not existing_terms:
            return 0.0
        
        query_terms_cf = self.get_collection_frequency(existing_terms)
        score = 0
        
        for term in existing_terms:
            tf = query_terms_tf[term]
            cf = query_terms_cf.get(term, 1)
            
            weight = math.log(1 + lambda_value/(1-lambda_value) * tf/single_doc_length * self.collection_N/cf)
            score += weight
        
        return score

    def build_trec_res_file(self, df, query_data, output_file):
        """Build TREC format result file"""
        print("🚀 Starting to build TREC format result file...")
        
        query_dict = dict(zip(query_data["qid"], query_data["query"]))
        print(f"📖 Loaded {len(query_dict)} queries")
        
        results = []
        total_pairs = len(df)
        start_time = time.time()
        processed = 0
        
        grouped = df.groupby("qid")
        
        for qid, group in grouped:
            if qid not in query_dict:
                print(f"⚠️ Query ID {qid} not found, skipping")
                continue
            
            query_text = query_dict[qid]
            doc_scores = []
            
            print(f"📝 Processing query {qid}: '{query_text[:50]}...'")
            
            for _, row in group.iterrows():
                docno = row["docno"]
                try:
                    score = self.lmjm_score(str(docno), query_text)
                    doc_scores.append((docno, score))
                except Exception as e:
                    print(f"⚠️ Document {docno} calculation failed: {e}")
                    doc_scores.append((docno, 0.0))
                
                processed += 1
                
                if processed % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_pairs - processed) / rate if rate > 0 else 0
                    print(f"📊 Progress: {processed}/{total_pairs} ({processed/total_pairs*100:.1f}%) "
                        f"Speed: {rate:.1f} docs/sec, ETA: {eta/60:.1f} min")
            
            for rank, (docno, score) in enumerate(doc_scores, 1):
                results.append(f"{qid}\tQ0\t{docno}\t{rank}\t{score:.6f}\tlmjm_run")
            
            print(f"✅ Query {qid} completed, {len(doc_scores)} documents")
        
        print(f"💾 Writing to file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in results:
                f.write(line + '\n')
        
        total_time = time.time() - start_time
        print(f"🎉 Completed!")
        print(f"📊 Statistics: Processed {processed} document pairs, took {total_time:.2f} seconds")
        print(f"📁 Output: {output_file}")
        
        return len(results)

    def simple_build_lmjm_res(self, basename, force_rebuild=False):
        """Simplified LMJM result file building function"""
        
        original_file = os.path.join(constants.trec_fair_res_folder, basename)
        if not os.path.exists(original_file):
            raise FileNotFoundError(f"❌ Original file does not exist: {original_file}")
        
        res_file = os.path.join(constants.trec_fair_res_folder, f"{basename}_lmjm.res")
        
        if os.path.exists(res_file) and not force_rebuild:
            file_size = os.path.getsize(res_file)
            with open(res_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f)
            print(f"✅ Result file already exists: {res_file}")
            print(f"   📊 Size: {file_size:,} bytes, {line_count:,} lines")
            return res_file
        
        print(f"📝 Starting build: {basename} -> {basename}_lmjm.res")
        
        df = pd.read_csv(original_file, sep="\t")
        df.rename(columns={"id": "qid", "page_id": "docno"}, inplace=True)
        
        query_data = pd.read_csv(constants.QUERIES_TREC_FAIR, sep="\t", header=None, names=["qid", "query"])
        
        self.build_trec_res_file(df, query_data, res_file)
        
        print(f"✅ Build completed: {res_file}")
        return res_file