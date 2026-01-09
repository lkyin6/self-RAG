# rag_engine.py
from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# =========================================================
# 1) 全局模型配置
# =========================================================
Settings.llm = Ollama(
    model="qwen2.5:7b",
    request_timeout=600.0,
    keep_alive=-1,
)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# =========================================================
# 2) 工具函数
# =========================================================
def _strip_code_fence(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()

def clean_json_obj(text: str) -> str:
    text = _strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return "{}"

# =========================================================
# 3) RAG 系统核心类
# =========================================================
class RAGSystem:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = self._load_documents(file_path)
        self.index: Optional[VectorStoreIndex] = None
        
        # 默认配置
        self.config: Dict[str, Any] = {
            "chunk_size": 256,
            "chunk_overlap": 20,
            "top_k": 3,
            "retrieve_k_multiplier": 4,
            "use_rerank": True,
            "similarity_cutoff": None,
        }

    def _load_documents(self, file_path: str) -> List[Document]:
        try:
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    text = "\n\n".join([f"{k}: {v}" for k, v in data.items()])
                elif isinstance(data, list):
                    text = "\n\n---\n\n".join([str(item) for item in data])
                else:
                    text = str(data)
                return [Document(text=text)]
            
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    return [Document(text=f.read())]

            return SimpleDirectoryReader(input_files=[file_path]).load_data()
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            return [Document(text="")]

    def build_index(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> None:
        if chunk_size is not None:
            self.config["chunk_size"] = int(chunk_size)
        if chunk_overlap is not None:
            self.config["chunk_overlap"] = int(chunk_overlap)

        print(f"🔄 重建索引: Chunk={self.config['chunk_size']}, Overlap={self.config['chunk_overlap']}")
        
        splitter = TokenTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
        )
        self.index = VectorStoreIndex.from_documents(
            self.documents, transformations=[splitter]
        )

    # ✅ 多线程打分 worker
    def _score_single_node(self, question, node_content, llm):
        prompt = f"""
你是检索重排序模型。请给片段打分 0-3（3=包含答案，0=无关）。
问题：{question}
片段：{node_content[:800]}

只输出JSON：{{"score":0}}
"""
        try:
            resp = llm.complete(prompt)
            obj = json.loads(clean_json_obj(str(resp)))
            return int(obj.get("score", 0))
        except:
            return 0

    def _llm_rerank(self, question: str, nodes: List[Any], rerank_llm, keep_k: int) -> List[Any]:
        if not nodes:
            return []
        
        # ✅ 使用 ThreadPoolExecutor 并发加速 (5线程)
        scores = [0] * len(nodes)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(
                    self._score_single_node, 
                    question, 
                    n.node.get_content() if hasattr(n, "node") else str(n), 
                    rerank_llm
                ): i 
                for i, n in enumerate(nodes)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                scores[idx] = future.result()

        # 组合 (score, node) 并排序
        scored_nodes = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)
        return [n for _, n in scored_nodes[:keep_k]]

    def query(
        self, 
        question: str, 
        top_k: Optional[int] = None, 
        llm=None, 
        retrieve_k: Optional[int] = None, 
        retrieve_k_multiplier: Optional[int] = None, 
        use_rerank: Optional[bool] = None, 
        rerank_llm=None, 
        similarity_cutoff: Optional[float] = None, 
        return_debug: bool = False
    ) -> Dict[str, Any]:
        
        # 参数覆盖
        cfg = self.config
        if top_k is not None: cfg["top_k"] = int(top_k)
        if retrieve_k_multiplier is not None: cfg["retrieve_k_multiplier"] = int(retrieve_k_multiplier)
        if use_rerank is not None: cfg["use_rerank"] = bool(use_rerank)
        if similarity_cutoff is not None: cfg["similarity_cutoff"] = float(similarity_cutoff)

        if self.index is None: self.build_index()

        # 1. 检索
        final_top_k = cfg["top_k"]
        mult = cfg.get("retrieve_k_multiplier", 4)
        pre_k = retrieve_k if retrieve_k else max(final_top_k, final_top_k * mult)
        
        retriever = self.index.as_retriever(similarity_top_k=pre_k)
        nodes = retriever.retrieve(question)
        raw_count = len(nodes)

        # 2. Cutoff 过滤
        cutoff = cfg.get("similarity_cutoff")
        if cutoff is not None:
            nodes = [n for n in nodes if (getattr(n, "score", 0) or 0) >= cutoff]
            # 兜底：如果过滤完没了，至少保留 1 个
            if not nodes and raw_count > 0: 
                nodes = retriever.retrieve(question)[:1]

        # 3. Rerank
        if cfg.get("use_rerank") and rerank_llm:
            nodes = self._llm_rerank(question, nodes, rerank_llm, final_top_k)
        else:
            nodes = nodes[:final_top_k]

        # 4. 生成
        contexts = [n.node.get_content() if hasattr(n, "node") else str(n) for n in nodes]
        ctx_block = "\n\n".join([f"[片段{i+1}]\n{c}" for i, c in enumerate(contexts)])
        
        gen_prompt = f"""
你是一个基于检索证据回答问题的助手。请只使用给定【检索片段】中的信息回答。
如果片段不足以回答，请明确说明“检索片段不足以回答”。

【问题】
{question}

【检索片段】
{ctx_block}

【回答】
"""
        try:
            ans = str((llm or Settings.llm).complete(gen_prompt))
        except Exception as e:
            ans = f"Error: {e}"

        res = {"answer": ans, "contexts": contexts}
        if return_debug:
            res["debug"] = {
                "final_top_k": final_top_k, 
                "raw_retrieved": raw_count,
                "after_filter": len(nodes)
            }
        return res