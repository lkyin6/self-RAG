# agent_logic.py
import json
import re
import time
from collections import Counter
from typing import List, Dict, Any, Optional

# LlamaIndex 核心组件
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# =========================================================
# 1. LLM 初始化 (带容错)
# =========================================================
try:
    mid_llm = Ollama(model="qwen2.5:1.5b", request_timeout=120.0)
except Exception as e:
    print(f"Warning: Failed to init qwen2.5:1.5b ({e}), using default Settings.llm")
    mid_llm = Settings.llm

try:
    fast_llm = Ollama(model="qwen2.5:0.5b", request_timeout=60.0)
except Exception:
    fast_llm = mid_llm

# =========================================================
# 2. 工具函数
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
# 3. 诊断与评估逻辑
# =========================================================
def diagnose_error(query: str, context_list: List[str], model_answer: str) -> Dict[str, Any]:
    context_text = "\n".join(context_list) if context_list else "(empty)"
    prompt = f"""
请判断基于【检索片段】生成的【AI回答】是否存在严重错误。
输入：
- 问：{query}
- 答：{model_answer}
- 片段：{context_text[:500]}

请返回 JSON 格式：
{{
    "error_type": "GOOD" | "MISSING_CONTENT" | "NOISE" | "SEGMENTATION_FAULT",
    "reason": "简短理由"
}}
"""
    try:
        response = fast_llm.complete(prompt)
        obj = json.loads(clean_json_obj(str(response)))
        
        et = str(obj.get("error_type", "GOOD")).upper()
        if "MISSING" in et: et = "MISSING_CONTENT"
        elif "NOISE" in et: et = "NOISE"
        elif "SEGMENT" in et: et = "SEGMENTATION_FAULT"
        else: et = "GOOD"
        
        return {"error_type": et, "reason": obj.get("reason", "ok")}
    except Exception:
        return {"error_type": "GOOD", "reason": "Diagnosis Parse Error"}

def eval_context_relevance(query: str, contexts: List[str]) -> Dict[str, Any]:
    if not contexts:
        return {"relevance": 0, "reason": "No contexts retrieved"}
    
    prompt = f"""
请对检索片段与问题的相关性打分 (0-3)。
0: 无关
1: 略微相关
2: 相关
3: 非常相关/包含答案

输入：
- 问：{query}
- 片段：{contexts[0][:500]}

返回 JSON: {{"relevance": 0, "reason": "..."}}
"""
    try:
        resp = mid_llm.complete(prompt)
        obj = json.loads(clean_json_obj(str(resp)))
        score = int(obj.get("relevance", 0))
        return {"relevance": score, "reason": obj.get("reason", "")}
    except Exception:
        return {"relevance": 0, "reason": "Relevance Parse Error"}

def rewrite_query(query: str) -> str:
    """如果检索效果不好，尝试改写查询"""
    try:
        prompt = f"请提取关键词并将以下问题改写为更适合检索的形式（只输出改写后的问题）：\n{query}"
        return str(mid_llm.complete(prompt)).strip()
    except Exception:
        return query

# =========================================================
# 4. 核心：生成测试题 (正则匹配版，最强鲁棒性)
# =========================================================
def generate_test_set(doc_text: str) -> List[Dict[str, str]]:
    sample = doc_text[:1000]
    if len(doc_text) > 3000:
        sample += "\n...\n" + doc_text[len(doc_text)//2 : len(doc_text)//2 + 1000]
    
    prompt = f"""
请根据以下文档内容，列出 5 个值得考核的关键问题。
不要包含答案。
每行一个问题。

文档片段：
{sample}
"""
    print("🤖 Agent 正在尝试生成问题...")
    try:
        response = str(mid_llm.complete(prompt))
        
        # 策略1: 抓取 "1. 问题" 或 "1、问题"
        pattern = r"\d+[\.\、]\s*(.*)"
        questions = re.findall(pattern, response)
        
        # 策略2: 如果没抓到，抓取带问号的行
        if not questions:
            questions = [l.strip() for l in response.split('\n') if ('?' in l or '？' in l) and len(l.strip()) > 5]
            
        # 策略3: 暴力抓取长句
        if not questions:
             questions = [l.strip() for l in response.split('\n') if len(l.strip()) > 8 and not l.strip().startswith(('-', '*', '#'))]

        final_set = []
        for q in questions:
            clean_q = re.sub(r"^[\-\*\#\>]\s*", "", q).strip()
            if len(clean_q) > 4:
                final_set.append({"question": clean_q, "standard_answer": "N/A"})
        
        if not final_set:
            print("⚠️ 自动生成失败，使用默认问题")
            return [{"question": "文档的主要内容是什么？", "standard_answer": "N/A"}]
            
        print(f"✅ 成功生成 {len(final_set[:5])} 个问题")
        return final_set[:5]

    except Exception as e:
        print(f"Generate Test Set Error: {e}")
        return [{"question": "文档主要讲了什么？", "standard_answer": "N/A"}]

# =========================================================
# 5. 核心：优化循环 (含最佳配置回溯)
# =========================================================
def run_optimization_loop(rag_system, test_set, status_container, doc_hint: str = ""):
    current_config = rag_system.config
    logs = []
    
    # 记录最佳状态
    best_score = -1.0
    best_config = rag_system.config.copy()

    for round_i in range(3):
        # 确保系统配置与当前策略同步
        rag_system.config.update(current_config)
        
        status_container.markdown(f"**Round {round_i+1}** (Chunk={current_config['chunk_size']}, TopK={current_config['top_k']}...)")
        
        round_errors = []
        rel_scores = []
        empty_cnt = 0

        for idx, qa in enumerate(test_set):
            q = qa["question"]
            if round_i > 0 and "rewritten" in qa: 
                q = qa["rewritten"]

            # 1. 查询
            try:
                res = rag_system.query(q, llm=mid_llm, rerank_llm=fast_llm, return_debug=True)
            except Exception as e:
                res = {"answer": f"Error: {e}", "contexts": []}

            contexts = res.get("contexts", [])
            if not contexts: 
                empty_cnt += 1
            
            # 2. 评估
            rel_obj = eval_context_relevance(q, contexts)
            diag = diagnose_error(q, contexts, res.get("answer", ""))
            
            rel_scores.append(rel_obj["relevance"])
            if diag["error_type"] != "GOOD": 
                round_errors.append(diag["error_type"])

            # 3. 日志
            logs.append({
                "round": round_i+1, 
                "question_id": idx+1, 
                "config": current_config.copy(),
                "relevance": rel_obj, 
                "diagnosis": diag,
                "inputs": {
                    "question": q, 
                    "contexts": contexts, 
                    "model_answer": res.get("answer","")
                }
            })

        # --- 决策 ---
        avg_rel = sum(rel_scores) / max(1, len(rel_scores))
        
        if avg_rel > best_score:
            best_score = avg_rel
            best_config = current_config.copy()
            status_container.success(f"📈 Score: {avg_rel:.2f} (New Best!)")
        else:
            status_container.warning(f"📉 Score: {avg_rel:.2f} (Best was {best_score:.2f})")

        if not round_errors and avg_rel >= 2.5:
            break

        # --- 调整 ---
        if empty_cnt > 1:
            current_config["top_k"] += 2
            status_container.info("Action: TopK +2 (Fix Empty Context)")
        elif avg_rel < 1.5:
            current_config["retrieve_k_multiplier"] = min(10, current_config.get("retrieve_k_multiplier", 4) + 2)
            current_config["use_rerank"] = True
            for qa in test_set: qa["rewritten"] = rewrite_query(qa["question"])
            status_container.info("Action: Rewrite Query & Increase Multiplier")
        elif "NOISE" in round_errors:
            current_config["similarity_cutoff"] = 0.3
            status_container.info("Action: Enable Cutoff 0.3")
        elif "SEGMENTATION_FAULT" in round_errors:
            current_config["chunk_overlap"] += 50
            rag_system.build_index()
            status_container.info("Action: Increase Overlap")
        else:
            current_config["top_k"] += 1
            status_container.info("Action: TopK +1")

    return logs, best_config