#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete index rebuilding tool
Supports both JSONL and TSV data sources for index construction
Improved version based on preliminary analysis results
"""

import os
import re
import json
import time
import psutil
import shutil
from pathlib import Path
from typing import Iterator, Dict, Optional, Tuple
import pyterrier as pt
import constants

class DocumentProcessor:
    """Improved document processor"""
    
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """
        Number normalization - fully matches Java implementation
        Java: content.replaceAll("(-)?\\d+(\\.\\d*)?", " _NUM_ ")
        """
        return re.sub(r'(-)?[0-9]+(\.[0-9]*)?', ' _NUM_ ', text)
    
    @staticmethod
    def process_document_content(title: str, plain: str) -> str:
        """
        Process document content - fully matches Java JSONDataIndexer logic:
        Java: String content = jsonLine.get("title").toString() + " " + jsonLine.get("plain").toString();
        then perform number normalization
        """
        content = f"{title} {plain}"
        content = DocumentProcessor.normalize_numbers(content)
        return content
    
    @staticmethod
    def process_tsv_content(content: str) -> str:
        """Process TSV format document content"""
        return DocumentProcessor.normalize_numbers(content)

class JSONLIndexBuilder:
    """JSONL format index builder"""
    
    def __init__(self,
                 collection_file: str = constants.trec_fair_orignal_data,
                 index_dir: str = constants.TREC_FAIR_INDEX_PATH,
                 stop_file: str = constants.STOPWORDS_FILE,
                 chunk_size: int = 50000):
        
        self.collection_file = os.path.abspath(collection_file)
        self.index_dir = os.path.abspath(index_dir)
        self.stop_file = os.path.abspath(stop_file)
        self.chunk_size = chunk_size
        
        self.threads = 1
        
        print(f"JSONL Index Builder initialized")
        print(f"Collection file: {self.collection_file}")
        print(f"Index directory: {self.index_dir}")
        print(f"Stopwords file: {self.stop_file}")
        
        self._check_files()
        
    
    def _check_files(self):
        """Check if necessary files exist"""
        if not os.path.exists(self.collection_file):
            raise FileNotFoundError(f"Collection file not found: {self.collection_file}")
        
        if not os.path.exists(self.stop_file):
            print(f"Stopwords file not found: {self.stop_file}")
            print("Will use PyTerrier default stopwords")
            self.stop_file = None
    
    def _init_pyterrier(self):
        """Initialize PyTerrier"""
        if not pt.java.started():
            print("Initializing PyTerrier...")
            pt.java.init()
    
    def load_stopwords(self) -> Optional[list]:
        """Load stopwords"""
        if not self.stop_file:
            return None
            
        print(f"Loading stopwords: {self.stop_file}")
        stopwords = []
        try:
            with open(self.stop_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.append(word)
            print(f"Loaded {len(stopwords)} stopwords")
            return stopwords
        except Exception as e:
            print(f"Failed to load stopwords: {e}")
            return None
    
    def document_generator(self) -> Iterator[Dict[str, str]]:
        """Efficient JSONL document generator"""
        print(f"Processing JSONL documents: {self.collection_file}")
        
        doc_count = 0
        error_count = 0
        normalized_count = 0
        duplicate_count = 0
        seen_ids = set()
        start_time = time.time()
        
        try:
            with open(self.collection_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        doc = json.loads(line)
                        
                        doc_id = doc.get('id', '')
                        if isinstance(doc_id, int):
                            doc_id = str(doc_id)
                        else:
                            doc_id = str(doc_id).strip()
                        
                        if not doc_id:
                            error_count += 1
                            continue
                        
                        if doc_id in seen_ids:
                            duplicate_count += 1
                            continue
                        seen_ids.add(doc_id)
                        
                        title = doc.get('title', '').strip()
                        plain = doc.get('plain', '').strip()
                        
                        original_content = f"{title} {plain}"
                        processed_content = DocumentProcessor.process_document_content(title, plain)
                        
                        if original_content != processed_content:
                            normalized_count += 1
                        
                        if not processed_content:
                            error_count += 1
                            continue
                        
                        yield {
                            "docno": str(doc_id),
                            "text": processed_content
                        }
                        
                        doc_count += 1
                        
                        if doc_count % self.chunk_size == 0:
                            elapsed = time.time() - start_time
                            rate = doc_count / elapsed
                            memory_mb = psutil.Process().memory_info().rss / (1024*1024)
                            
                            print(f"Processed {doc_count:,} documents "
                                 f"({rate:.0f} docs/sec, memory: {memory_mb:.0f}MB)")
                            print(f"   Stats: duplicates: {duplicate_count:,}, normalized: {normalized_count:,}")
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        error_count += 1
                        if error_count <= 10:
                            print(f"Skipping line {line_num}: {e}")
                        continue
        
        except Exception as e:
            print(f"Document generator error: {e}")
            raise
        
        elapsed = time.time() - start_time
        final_rate = doc_count / elapsed if elapsed > 0 else 0
        print(f"JSONL document processing completed:")
        print(f"   Total documents: {doc_count:,}")
        print(f"   Normalized documents: {normalized_count:,}")
        print(f"   Duplicate documents: {duplicate_count:,}")
        print(f"   Error documents: {error_count:,}")
        print(f"   Average speed: {final_rate:.0f} docs/sec")
    
    def build_index(self):
        """Build JSONL index"""
        print("Starting JSONL index construction...")
        
        stopwords = self.load_stopwords()
        
        if Path(self.index_dir).exists():
            raise FileExistsError(f"Index directory already exists: {self.index_dir}")
        
        indexer_params = {
            'index_path': self.index_dir,
            'verbose': True,
            'stemmer': 'porter',
            'type': pt.IndexingType.CLASSIC,
            'tokeniser': "english",
            'text_attrs': ["text"],
            'meta_reverse': ["docno"],
            'meta': {
                'docno': 20,
                'text': 100000
            }
        }
        
        if stopwords:
            indexer_params['stopwords'] = stopwords
        
        indexer = pt.IterDictIndexer(**indexer_params)
        
        print("Starting index construction...")
        start_time = time.time()
        
        try:
            index_ref = indexer.index(self.document_generator())
            
            build_time = time.time() - start_time
            print(f"JSONL index construction successful!")
            print(f"Build time: {build_time:.2f} seconds")
            print(f"Index location: {self.index_dir}")
            
            self._verify_index(index_ref)
            
            return index_ref
            
        except Exception as e:
            print(f"Index construction failed: {e}")
            raise
    
    def _verify_index(self, index_ref):
        """Verify index"""
        try:
            print("Verifying index...")
            index = pt.IndexFactory.of(index_ref)
            stats = index.getCollectionStatistics()
            
            print(f"Index verification successful:")
            print(f"   Documents: {stats.getNumberOfDocuments():,}")
            print(f"   Unique terms: {stats.getNumberOfUniqueTerms():,}")
            print(f"   Total tokens: {stats.getNumberOfTokens():,}")
            print(f"   Average document length: {stats.getAverageDocumentLength():.2f}")
            
        except Exception as e:
            print(f"Index verification failed: {e}")

class TSVIndexBuilder:
    """TSV format index builder"""
    
    def __init__(self, 
                 collection_file: str = constants.MSMARCO_Original_data,
                 index_dir: str = constants.MSMARCO_INDEX_PATH,
                 stop_file: str = constants.STOPWORDS_FILE):
        
        self.collection_file = os.path.abspath(collection_file)
        self.index_dir = os.path.abspath(index_dir)
        self.stop_file = os.path.abspath(stop_file)
        
        print(f"TSV Index Builder initialized")
        print(f"Collection file: {self.collection_file}")
        print(f"Index directory: {self.index_dir}")
        
        if not pt.java.started():
            print("Initializing PyTerrier...")
            pt.java.init()

    def _monitor_progress(self, total_start_time):
        """Background monitoring process"""
        while True:
            time.sleep(30)
            elapsed = time.time() - total_start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Progress monitor - Running: {elapsed:.1f}s, Memory: {memory_usage:.1f}MB")
    
    def tsv_generate(self, limit=None):
        """TSV document generator with progress monitoring"""
        count = 0
        start_time = time.time()
        last_update_time = start_time
        update_interval = 50000
        
        print("Reading TSV documents...")
        
        with pt.io.autoopen(self.collection_file, 'rt') as corpusfile:
            for line in corpusfile:
                if limit and count >= limit:
                    break
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                docno, passage = parts
                yield {'docno': docno, 'text': DocumentProcessor.normalize_numbers(passage)}
                count += 1
                
                if count % update_interval == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    interval_time = current_time - last_update_time
                    docs_per_sec = update_interval / interval_time
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    print(f"   Processed {count:,} documents")
                    print(f"      Total time: {elapsed:.1f}s, Recent {update_interval:,} docs: {interval_time:.1f}s")
                    print(f"      Processing speed: {docs_per_sec:.0f} docs/sec")
                    print(f"      Memory usage: {memory_usage:.1f}MB")
                    
                    last_update_time = current_time
        
        total_time = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        avg_speed = count / total_time if total_time > 0 else 0
        
        print(f"TSV document reading completed:")
        print(f"   Total documents: {count:,}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average speed: {avg_speed:.0f} docs/sec")
        print(f"   Final memory: {final_memory:.1f}MB")
    
    def build_index(self):
        """Build TSV index with detailed monitoring"""
        print("Starting TSV index construction...")
        
        total_start_time = time.time()
        
        if not os.path.exists(self.collection_file):
            raise FileNotFoundError(f"Collection file not found: {self.collection_file}")
        
        if os.path.exists(self.index_dir):
            raise FileExistsError(f"Index directory already exists: {self.index_dir}")
        
        print("Loading stopwords...")
        stopwords_start = time.time()
        stopwords = None
        if os.path.exists(self.stop_file):
            with open(self.stop_file, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f if line.strip()]
        stopwords_time = time.time() - stopwords_start
        print(f"Stopwords loaded: {len(stopwords) if stopwords else 0} words, time: {stopwords_time:.2f}s")
        
        print("Creating indexer...")
        indexer_start = time.time()
        indexer = pt.IterDictIndexer(
            index_path=self.index_dir,
            meta={"docno": 200, "text": 10000},
            overwrite=True,
            text_attrs=["text"],
            meta_reverse=["docno"],
            stopwords=stopwords,
            tokeniser="english",
            stemmer="porter",
        )
        indexer_time = time.time() - indexer_start
        print(f"Indexer created, time: {indexer_time:.2f}s")
        
        print("Starting index construction process...")
        print("Index construction stages:")
        print("   1. Document reading and preprocessing")
        print("   2. Term extraction and tokenization")
        print("   3. Inverted index construction")
        print("   4. Index compression and optimization")
        
        index_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Memory usage before indexing: {initial_memory:.1f}MB")
        
        try:
            print("Starting data indexing...")
            index_ref = indexer.index(self.tsv_generate())
            
            index_time = time.time() - index_start
            total_time = time.time() - total_start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            print(f"TSV index construction successful!")
            print(f"Construction statistics:")
            print(f"   Index construction time: {index_time:.2f} seconds")
            print(f"   Total construction time: {total_time:.2f} seconds")
            print(f"   Memory increase: {memory_increase:+.1f}MB ({initial_memory:.1f}→{final_memory:.1f}MB)")
            print(f"Index location: {self.index_dir}")
            
            if os.path.exists(self.index_dir):
                index_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(self.index_dir)
                               for filename in filenames) / 1024 / 1024
                print(f"Index file size: {index_size:.1f}MB")
            
            return index_ref
            
        except Exception as e:
            failed_time = time.time() - total_start_time
            failed_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"TSV index construction failed: {e}")
            print(f"Runtime before failure: {failed_time:.2f} seconds")
            print(f"Memory usage at failure: {failed_memory:.1f}MB")
            raise

    def _verify_index(self, index_ref):
        """Verify index with performance monitoring"""
        try:
            print("Verifying index...")
            verify_start = time.time()
            verify_memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            index = pt.IndexFactory.of(index_ref)
            stats = index.getCollectionStatistics()
            
            verify_time = time.time() - verify_start
            verify_memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = verify_memory_end - verify_memory_start
            
            print(f"Index verification successful:")
            print(f"Index statistics:")
            print(f"   Documents: {stats.getNumberOfDocuments():,}")
            print(f"   Unique terms: {stats.getNumberOfUniqueTerms():,}")
            print(f"   Total tokens: {stats.getNumberOfTokens():,}")
            print(f"   Average document length: {stats.getAverageDocumentLength():.2f}")
            print(f"Verification performance:")
            print(f"   Verification time: {verify_time:.2f} seconds")
            print(f"   Verification memory overhead: {memory_used:+.1f}MB")
            
        except Exception as e:
            print(f"Index verification failed: {e}")

def main():
    """Main function - rebuild all indexes"""
    print("Index Rebuilding Tool")
    print("=" * 50)
    
    try:
        # 1. Build JSONL index
        print("\nStep 1: Build JSONL index")
        print("-" * 30)
        
        jsonl_builder = JSONLIndexBuilder(
            collection_file=constants.trec_fair_orignal_data,
            index_dir=constants.TREC_FAIR_INDEX_PATH,
            stop_file=constants.STOPWORDS_FILE
        )
        
        jsonl_index = jsonl_builder.build_index()
        
        # 2. Build TSV index
        print("\nStep 2: Build TSV index")
        print("-" * 30)
        
        tsv_builder = TSVIndexBuilder(
            collection_file=constants.MSMARCO_Original_data,
            index_dir=constants.MSMARCO_INDEX_PATH,
            stop_file=constants.STOPWORDS_FILE
        )
        
        tsv_index = tsv_builder.build_index()
        tsv_builder._verify_index(tsv_index)
        print("\nAll indexes rebuilt successfully!")
        print("Available indexes:")
        print(f"   JSONL index: {constants.TREC_FAIR_INDEX_PATH}")
        # print(f"   TSV index: {constants.MSMARCO_INDEX_PATH}")
    except Exception as e:
        print(f"\nIndex rebuilding failed: {e}")
        raise

if __name__ == "__main__":
    main() 