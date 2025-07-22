"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506060435
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

from .dataType import DocChunk, lookupQuery
from .templates import prompt_tem, query_tem, lm_tem

class RAGFramework:
    def __init__(self, 
        lm_model_name: str = "gemma-3-4b-it", 
        emb_model_name: str = "all-MiniLM-L6-v2",
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

        self.index = None
        self.documents = []

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
        
        # Generate embedding vectors
        contents = [doc.content for doc in all_chunks]
        embeddings = self.embedding_model.encode(contents)
        self.index = faiss.IndexFlatIP(embeddings.shape[1]) # Create FAISS index
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        
        # Save documents
        for doc, embedding in zip(all_chunks, embeddings):
            doc.embedding = embedding
        self.chunk_list = all_chunks
        print(f"Successfully loaded {len(self.chunk_list)} document chunks")
    
    def retrieve_relevant_docs(self, 
        search_query: str, 
        top_k: int = 3
    ) -> List[DocChunk]:
        query_embedding = self.embedding_model.encode([search_query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_list):
                doc = self.chunk_list[idx]
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
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
        
        docs_data = []
        for doc in self.chunk_list:
            docs_data.append({
                'idx': doc.idx,
                'url': doc.url,
                'content': doc.content,
            })
        
        with open(os.path.join(save_path, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        print(f"Save index to {save_path}")
    
    def load_index(self, 
        load_path: str
    ):
        self.index = faiss.read_index(os.path.join(load_path, "index.faiss"))
        
        with open(os.path.join(load_path, "documents.json"), 'r', encoding='utf-8') as f:
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
            device=cfg.get("device", "cuda"),
        )

        if "doc_path" in cfg:
            doc_path = cfg["doc_path"]
            rag.load_doc_from_path(doc_path)
            rag.save_index("./ref_idx/")

        if "idx_path" in cfg:
            idx_path = cfg["idx_path"]
            rag.load_index(doc_path)
        print("RAGFramework initialized\n")

        return rag












