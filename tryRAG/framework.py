"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507231725
"""


import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import Counter
import math
import matplotlib.pyplot as plt

from .dataType import DocChunk, lookupQuery
from .templates import prompt_tem, query_tem, lm_tem, preans_tem

class RAGFramework:
    def __init__(self, 
        lm_model_name: str = "gemma-3-4b-it", 
        emb_model_name: str = "all-MiniLM-L6-v2", 
        mode: str = "dense", #@ "dense" / "sparse" / "hybrid"
        chunk_level: str = "paragraph", #@ "web_page" / "paragraph" / "sentence"
        more_info: bool = False, 
        device="cuda", 
    ):
        self.device = device
        if mode == 'hybrid':
            self.mode = ["dense", "sparse"]
        else:
            self.mode = [mode]
        self.chunk_level = chunk_level
        self.more_info = more_info
        
        if "dense" in self.mode:
            self.embedding_model = SentenceTransformer(emb_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            lm_model_name,
            # torch_dtype=torch.float16,
            device_map="auto",
            # trust_remote_code=True
        ).to(self.device)

        self.index = None
        self.documents = []
        self.bm25 = BM25()

    def preprocess_pages2chunks(self, 
        pages_list, 
    ):
        chunked_list = []
        for page in pages_list:

            if self.chunk_level == 'web_page':
                chunked_list.append({
                    'text': page['text'].strip(),
                    'upper_text': None, 
                    'url': page['url']
                })

            paragraphs = page['text'].split('\n\n')
            for para_idx in range(len(paragraphs)):

                if self.chunk_level == 'paragraph':
                    if self.more_info:
                        if len(paragraphs) == 1:
                            text = paragraphs[para_idx].strip()
                        elif para_idx == 0:
                            text = f"{paragraphs[para_idx].strip()}\n\n{paragraphs[para_idx + 1].strip()}"
                        elif para_idx == len(paragraphs) - 1:
                            text = f"{paragraphs[para_idx - 1].strip()}\n\n{paragraphs[para_idx].strip()}"
                        else:
                            text = f"{paragraphs[para_idx - 1].strip()}\n\n{paragraphs[para_idx].strip()}\n\n{paragraphs[para_idx + 1].strip()}"

                        chunked_list.append({
                            'text': text,
                            'upper_text': page['text'].strip(), 
                            'url': page['url']
                        })
                    else:
                        chunked_list.append({
                            'text': paragraphs[para_idx].strip(),
                            'upper_text': page['text'].strip(), 
                            'url': page['url']
                        })
                elif self.chunk_level == 'sentence':
                    sentences = re.split(r'(?<=[.!?]) +', paragraphs[para_idx].strip())
                    for sentence in sentences:
                        chunked_list.append({
                            'text': sentence.strip(),
                            'upper_text': paragraphs[para_idx].strip(), 
                            'url': page['url']
                        })

        idx = 0
        all_chunks = []
        token_len_list = []
        for chunked in chunked_list:
            token_len = self.tokenizer(
                chunked['text'], 
                return_tensors='pt', 
                truncation=True, 
                max_length=2048, 
            )['input_ids'].shape[-1]
            token_len = int(token_len)

            doc = DocChunk(
                idx=idx, 
                url=chunked['url'], 
                content=chunked['text'], 
                upper_text=chunked['upper_text'],
                token_len=token_len,
            )
            token_len_list += [token_len]
            all_chunks += [doc]
            idx += 1

        data = np.asarray(token_len_list)
        plt.title(self.chunk_level)
        plt.hist(data, bins=100)
        plt.show()

        print(f"avg: {sum(token_len_list) / len(token_len_list)} tokens per chunk")
        print(f"max: {max(token_len_list)} tokens per chunk")
        print(f"hit: {token_len_list.count(2048)} / {len(token_len_list)}")
        return all_chunks
    
    def load_doc_from_path(self, 
        documents_path: str
    ):
        pages_list = []
        with open(documents_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    pages_list.append(data)

        all_chunks = self.preprocess_pages2chunks(pages_list)
        contents = [doc.content for doc in all_chunks]

        if "dense" in self.mode:
            embeddings = self.embedding_model.encode(contents)
            self.index = faiss.IndexFlatIP(embeddings.shape[1]) # Create FAISS index
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype(np.float32))
            
            # Save documents
            for doc, embedding in zip(all_chunks, embeddings):
                doc.embedding = embedding

        if "sparse" in self.mode:
            self.bm25.fit(contents)

        self.chunk_list = all_chunks
        print(f"Successfully loaded {len(self.chunk_list)} document chunks")
    
    def retrieve_relevant_docs(self, 
        search_query: str, 
        top_k: int = 3
    ) -> List[DocChunk]:
        
        relevant_docs = []
        if "dense" in self.mode:
            query_embedding = self.embedding_model.encode([search_query])
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_list):
                    doc = self.chunk_list[idx]
                    relevant_docs.append(doc)

        if "sparse" in self.mode:
            results = self.bm25.search(search_query, top_k)

            for score, idx in results:
                if idx < len(self.chunk_list):
                    doc = self.chunk_list[idx]
                    doc.score = score
                    relevant_docs.append(doc)

        seen = set()
        unique_data = []
        for doc in relevant_docs:
            if doc.idx not in seen:
                seen.add(doc.idx)
                unique_data.append(doc)

        return unique_data

    def _generate(self, 
        text: str,
    ):
        prompt = lm_tem(
            text=text
        )

        inputs = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )#.to(self.model.device, dtype=torch.bfloat16)

        inputs = {
            k: (
                v.to(self.model.device, dtype=torch.bfloat16)
                if v.dtype.is_floating_point else v.to(self.model.device)
            )
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]

        max_len = int(self.model.config.text_config.max_position_embeddings)
        if input_len > max_len:
            raise ValueError(
                f"Input length {input_len} exceeds maximum allowed length of {max_len} tokens."
            )

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True
            )
            generation = generation[0][input_len:]
        response = self.tokenizer.decode(generation, skip_special_tokens=True)
        return response

    def generate(self, 
        query: lookupQuery, 
        top_k: int = 3, 
        use_upper_text: bool = False
    ) -> str:
        search_query = query_tem(
            query=query, 
        )
        relevant_docs = self.retrieve_relevant_docs(
            search_query=search_query, 
            top_k=top_k, 
        )

        if use_upper_text:
            seen = set()
            unique_data = []
            for doc in relevant_docs:
                if doc.upper_text not in seen:
                    seen.add(doc.upper_text)
                    unique_data.append(doc)
            relevant_docs = unique_data

        context = ""
        for i, doc in enumerate(relevant_docs):
            if use_upper_text:
                context += f"References {i+1}:{doc.upper_text}\n"
            else:
                context += f"References {i+1}:{doc.content}\n"
        
        text = prompt_tem(
            context=context, 
            query=query, 
        )
        response = self._generate(text=text)

        return {
            'response': response, 
            'prompt': text, 
            'relevant_docs': relevant_docs, 
        }

    def ask(self, 
        question: str, 
        top_k: int = 3, 
        use_upper_text: bool = False, 
        pre_answer: bool = False, 
    ):
        if pre_answer:
            text = preans_tem(question=question)
            response = self._generate(text=text)
            response = response.split('Answer:')[-1]
            preansing = f"{question}\nSpeculation:\n{response}"
            query = lookupQuery(
                question=preansing, 
            )
        else:
            query = lookupQuery(
                question=question, 
            )
            
        response = self.generate(
            query=query, 
            top_k=top_k,
            use_upper_text=use_upper_text
        )
        if pre_answer:
            response['pre_answer'] = preansing
        return response

    def save_index(self, 
        save_path: str
    ):
        os.makedirs(save_path, exist_ok=True)
        
        if "dense" in self.mode:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(save_path, f"dense_index.faiss"))

        if "sparse" in self.mode:
            # Save BM25 index
            bm25_data = {
                'k1': self.bm25.k1,
                'b': self.bm25.b,
                'doc_freqs': self.bm25.doc_freqs,
                'idf': self.bm25.idf,
                'doc_len': self.bm25.doc_len,
                'avgdl': self.bm25.avgdl,
                'corpus': self.bm25.corpus
            }
            with open(os.path.join(save_path, f"sparse_index.json"), 'w', encoding='utf-8') as f:
                json.dump(bm25_data, f)
        
        # Save documents data
        docs_data = []
        for doc in self.chunk_list:
            doc_dict = {
                'idx': doc.idx,
                'url': doc.url,
                'content': doc.content,
            }
            # Save embedding for dense mode
            if "dense" in self.mode and hasattr(doc, 'embedding'):
                doc_dict['embedding'] = doc.embedding.tolist()
            docs_data.append(doc_dict)
        
        with open(os.path.join(save_path, f"documents.json"), 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {self.mode} index to {save_path}")
    
    def load_index(self, 
        load_path: str
    ):
        if "dense" in self.mode:
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(load_path, f"dense_index.faiss"))

        if "sparse" in self.mode:
            # Load BM25 index
            with open(os.path.join(load_path, f"sparse_index.json"), 'r', encoding='utf-8') as f:
                bm25_data = json.load(f)
            
            self.bm25 = BM25(k1=bm25_data['k1'], b=bm25_data['b'])
            self.bm25.doc_freqs = bm25_data['doc_freqs']
            self.bm25.idf = bm25_data['idf']
            self.bm25.doc_len = bm25_data['doc_len']
            self.bm25.avgdl = bm25_data['avgdl']
            self.bm25.corpus = bm25_data['corpus']
        
        # Load documents
        with open(os.path.join(load_path, f"documents.json"), 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        self.chunk_list = []
        for doc_data in docs_data:
            doc = DocChunk(
                idx=doc_data['idx'],
                url=doc_data['url'],
                content=doc_data['content'],
            )
            # Restore embedding for dense mode
            if "dense" in self.mode and 'embedding' in doc_data:
                doc.embedding = np.array(doc_data['embedding'])
            self.chunk_list.append(doc)
        
        print(f"Loaded {self.mode} index from {load_path}")

    @classmethod
    def from_config(cls, 
        cfg
    ):
        rag = cls(
            lm_model_name=cfg.get("lm_model_name", "gemma-3-4b-it"),
            emb_model_name=cfg.get("emb_model_name", "all-MiniLM-L6-v2"),
            mode=cfg.get("mode", "dense"),  # "dense" or "sparse"
            chunk_level=cfg.get("chunk_level", "paragraph"),  # "web_page" / "paragraph" / "sentence"
            more_info=cfg.get("more_info", False),
            device=cfg.get("device", "cuda"),
        )

        if "doc_path" in cfg:
            doc_path = cfg["doc_path"]
            rag.load_doc_from_path(doc_path)
            rag.save_index("./ref_idx/")

        if "idx_path" in cfg:
            idx_path = cfg["idx_path"]
            rag.load_index(idx_path)
        print("RAGFramework initialized\n")

        return rag

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
    def tokenize(self, text):
        """Simple tokenization - split on whitespace and punctuation"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, corpus):
        """Build BM25 index from corpus"""
        self.corpus = corpus
        self.doc_len = []
        self.doc_freqs = []
        
        # Tokenize all documents and compute term frequencies
        for doc in corpus:
            tokens = self.tokenize(doc)
            self.doc_len.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate IDF for each term
        self.idf = {}
        all_terms = set()
        for freq in self.doc_freqs:
            all_terms.update(freq.keys())
        
        for term in all_terms:
            containing_docs = sum(1 for freq in self.doc_freqs if term in freq)
            self.idf[term] = math.log((len(corpus) - containing_docs + 0.5) / (containing_docs + 0.5) + 1.0)
    
    def search(self, query, top_k=10):
        """Search for most relevant documents"""
        query_tokens = self.tokenize(query)
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[i]
            
            for token in query_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append((score, i))
        
        # Sort by score (descending) and return top_k indices
        scores.sort(key=lambda x: x[0], reverse=True)
        return [(score, idx) for score, idx in scores[:top_k]]











