# app.py
import streamlit as st
import os
import re
from collections import defaultdict

# 确保引用正确
from agent_logic import generate_test_set, run_optimization_loop, mid_llm 
from rag_engine import RAGSystem

st.set_page_config(page_title="RAG 自动调优 Agent", layout="wide")
st.title("🤖 RAG Hyper-Optimizer Agent")

# =========================================================
# 0) 纯 Python 工具：完全替代 Pandas/Numpy
# =========================================================
def rows_to_markdown(rows: list[dict]) -> str:
    if not rows: return "无数据"
    headers = list(rows[0].keys())
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        row_str = []
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float): val = f"{val:.2f}"
            row_str.append(str(val))
        md += "| " + " | ".join(row_str) + " |\n"
    return md

# =========================================================
# 1) 指标计算工具
# =========================================================
def compute_rag_metrics_per_round(logs, judge_llm):
    by_r = defaultdict(list)
    for x in logs: by_r[x["round"]].append(x)
    out = {}
    for r, items in sorted(by_r.items()):
        rel_scores = [float(x.get("relevance",{}).get("relevance",0)) for x in items]
        avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0
        
        faithful_scores = [1.0 if x.get("diagnosis",{}).get("error_type")=="GOOD" else 0.0 for x in items]
        avg_faith = sum(faithful_scores) / len(faithful_scores) if faithful_scores else 0
        
        out[r] = {
            "Round": r, 
            "Relevance": avg_rel, 
            "Faithfulness": avg_faith,
            "Count": len(items)
        }
    return out

# =========================================================
# 2) UI 主逻辑
# =========================================================
with st.sidebar:
    st.header("1. 上传数据")
    uploaded_file = st.file_uploader("文件上传", type=["pdf", "json", "txt"])
    debug_mode = st.checkbox("默认展开详细日志", value=False)
    
    st.divider()
    st.header("2. 考题设置")
    # ✅ 功能新增：用户选择
    question_mode = st.radio(
        "考题来源：",
        ("🤖 Agent 自动生成", "✍️ 用户手动输入"),
        index=0
    )

if uploaded_file:
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, uploaded_file.name)
    with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    # RAG 初始化
    if "rag_system" not in st.session_state:
        try:
            with st.spinner("正在初始化 RAG 引擎..."):
                st.session_state.rag_system = RAGSystem(save_path)
                st.session_state.rag_system.build_index()
            st.success("引擎初始化完成！")
        except Exception as e:
            st.error(f"系统初始化失败: {e}")
            st.stop()
    
    rag_system = st.session_state.rag_system

    # =================================================
    # 处理考题逻辑
    # =================================================
    test_set = []
    
    if question_mode == "✍️ 用户手动输入":
        st.info("请在下方输入你想测试的问题，Agent 将针对这些问题优化 RAG 参数。")
        user_input = st.text_area(
            "输入测试问题（每行一个）：", 
            height=150,
            placeholder="例如：\n这里的 Chunk Size 是多少？\n文档提到的核心算法是什么？"
        )
        if user_input.strip():
            lines = [line.strip() for line in user_input.split('\n') if line.strip()]
            test_set = [{"question": line, "standard_answer": "N/A"} for line in lines]
            st.caption(f"✅ 已识别 {len(test_set)} 个自定义问题")
    
    else:
        st.info("点击启动后，Agent 将阅读文档并尝试自动生成考题。")

    # =================================================
    # 启动按钮
    # =================================================
    if st.button("🚀 启动 Agent 自动调优"):
        
        # 1. 自动生成模式
        if question_mode == "🤖 Agent 自动生成":
            with st.status("正在阅读文档并生成考题...", expanded=True) as s:
                doc_text = rag_system.documents[0].text if rag_system.documents else ""
                # 调用正则修复版生成函数
                test_set = generate_test_set(doc_text) 
                s.update(label=f"成功生成 {len(test_set)} 道考题", state="complete")
        
        # 2. 检查
        if not test_set:
            st.error("❌ 未检测到测试题。如果是手动模式，请先在输入框中填写问题。")
            st.stop()

        with st.expander("👀 查看当前测试集", expanded=True):
            for i, qa in enumerate(test_set):
                st.text(f"{i+1}. {qa['question']}")

        # 3. 运行优化
        st.subheader("🛠️ 优化过程监控")
        log_container = st.container()
        
        try:
            logs, final_config = run_optimization_loop(
                rag_system, test_set, log_container
            )
        except Exception as e:
            st.error(f"优化过程出错: {e}")
            st.stop()
        
        # 4. 结果展示
        st.divider()
        st.subheader("📈 优化结果")
        metrics = compute_rag_metrics_per_round(logs, judge_llm=mid_llm)
        rows = [metrics[r] for r in sorted(metrics.keys())]
        st.markdown(rows_to_markdown(rows))
        
        st.success("🏆 推荐最佳配置")
        st.json(final_config)
        
        st.divider()
        st.subheader("📝 详细日志")
        for log in logs:
            title = f"R{log['round']} | {log['inputs']['question']} | {log['diagnosis']['error_type']}"
            with st.expander(title, expanded=debug_mode):
                st.markdown(f"**Answer:** {log['inputs']['model_answer']}")
                st.markdown(f"**Reason:** {log['diagnosis']['reason']}")
                st.markdown("**Contexts:**")
                for c in log['inputs']['contexts']:
                    st.text(c[:200]+"...")

else:
    st.info("👈 请先上传文档。")