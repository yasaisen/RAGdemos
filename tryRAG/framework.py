"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507221631
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
import json
from collections import Counter
import math

from .dataType import DocChunk, lookupQuery
from .templates import prompt_tem, query_tem, lm_tem

class RAGFramework:
    def __init__(self, 
        lm_model_name: str = "gemma-3-4b-it", 
        emb_model_name: str = "all-MiniLM-L6-v2", 
        mode: str = "dense ", #@ "dense" / "sparse"
        device="cuda", 
    ):
        self.device = device
        self.embedding_model = SentenceTransformer(emb_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            lm_model_name,
            # torch_dtype=torch.float16,
            device_map="auto",
            # trust_remote_code=True
        ).to(self.device)

        self.mode = mode
        self.index = None
        self.documents = []
        self.bm25 = BM25()

    def preprocess_text2chunk(self, 
        text: str
    ) -> List[str]:
        chunks = text
        return chunks
    
    def load_doc_from_path(self, 
        documents_path: str
    ):
        all_chunks = []
        idx = 0
        with open(documents_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    chunks = self.preprocess_text2chunk(data['text'])
                    doc = DocChunk(
                        idx=idx, 
                        url=data['url'], 
                        content=chunks, 
                    )
                    all_chunks += [doc]
                    idx += 1
        
        contents = [doc.content for doc in all_chunks]

        if self.mode == "dense":
            embeddings = self.embedding_model.encode(contents)
            self.index = faiss.IndexFlatIP(embeddings.shape[1]) # Create FAISS index
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype(np.float32))
            
            # Save documents
            for doc, embedding in zip(all_chunks, embeddings):
                doc.embedding = embedding

        elif self.mode == "sparse":
            self.bm25.fit(contents)

        self.chunk_list = all_chunks
        print(f"Successfully loaded {len(self.chunk_list)} document chunks")
    
    def retrieve_relevant_docs(self, 
        search_query: str, 
        top_k: int = 3
    ) -> List[DocChunk]:
        if self.mode == "dense":
            query_embedding = self.embedding_model.encode([search_query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            relevant_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_list):
                    doc = self.chunk_list[idx]
                    relevant_docs.append(doc)

        elif self.mode == "sparse":

            results = self.bm25.search(search_query, top_k)

            relevant_docs = []
            for score, idx in results:
                if idx < len(self.chunk_list):
                    doc = self.chunk_list[idx]
                    # Store the BM25 score in the document object for potential use
                    doc.score = score
                    relevant_docs.append(doc)

        return relevant_docs

    def generate(self, 
        query: lookupQuery, 
        top_k: int = 3, 
    ) -> str:
        search_query = query_tem(
            query=query, 
        )
        relevant_docs = self.retrieve_relevant_docs(
            search_query=search_query, 
            top_k=top_k, 
        )
        
        context = ""
        for i, doc in enumerate(relevant_docs):
            context += f"References {i+1}:{doc.content}\n"
        
        text = prompt_tem(
            context=context, 
            query=query, 
        )
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

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True
            )
            generation = generation[0][input_len:]
        response = self.tokenizer.decode(generation, skip_special_tokens=True)
        return {
            'response': response, 
            'prompt': text, 
            'relevant_docs': relevant_docs, 
        }

    def ask(self, 
        question: str
    ):
        query = lookupQuery(
            question=question, 
        )
        response = self.generate(
            query=query
        )
        return response

    def save_index(self, 
        save_path: str
    ):
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_path, f"{self.mode}_index.faiss"))
        
        docs_data = []
        for doc in self.chunk_list:
            docs_data.append({
                'idx': doc.idx,
                'url': doc.url,
                'content': doc.content,
            })
        
        with open(os.path.join(save_path, f"{self.mode}_documents.json"), 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        print(f"Save index to {save_path}")
    
    def load_index(self, 
        load_path: str
    ):
        self.index = faiss.read_index(os.path.join(load_path, "index.faiss"))
        
        with open(os.path.join(load_path, "{self.mode}_documents.json"), 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        self.chunk_list = []
        for doc_data in docs_data:
            doc = DocChunk(
                idx=doc_data['idx'],
                url=doc_data['url'],
                content=doc_data['content'],
            )
            self.chunk_list.append(doc)
        
        print(f"Load from {load_path}")

    @classmethod
    def from_config(cls, 
        cfg
    ):
        rag = cls(
            lm_model_name=cfg.get("lm_model_name", "gemma-3-4b-it"),
            emb_model_name=cfg.get("emb_model_name", "all-MiniLM-L6-v2"),
            mode=cfg.get("mode", "dense"),  # "dense" or "sparse"
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











