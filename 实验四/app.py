# app.py - 适配 ChromaDB 版本（小改优化）
import streamlit as st
import time
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'D:\.cache\huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:\.cache\huggingface'

# 导入配置和模块
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, COLLECTION_NAME, id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
# 导入 ChromaDB 工具函数
from chroma_utils import get_chroma_client, setup_chroma_collection, search_similar_documents
from rag_core import generate_answer

# --- Streamlit UI 设置 ---
st.set_page_config(
    page_title="医疗智能问答系统",
    page_icon="🏥",
    layout="wide"
)

# 添加简单的CSS样式（不会冲突）
st.markdown("""
<style>
    /* 简单的卡片样式 */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin: 10px 0;
    }

    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* 响应式调整 */
    @media (max-width: 768px) {
        .stTextInput>div>div>input {
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 医疗智能问答系统")
st.markdown(f"**嵌入模型**: `{EMBEDDING_MODEL_NAME}` | **生成模型**: `{GENERATION_MODEL_NAME}`")

# --- 系统状态显示 ---
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("检索数量", TOP_K)
with status_col2:
    st.metric("最大索引", MAX_ARTICLES_TO_INDEX)
with status_col3:
    # 尝试显示数据量
    try:
        import json

        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            st.metric("数据条目", len(data))
    except:
        st.metric("数据状态", "加载中...")

st.divider()

# --- 初始化与缓存 ---
# 获取 ChromaDB 客户端
chroma_client = get_chroma_client()

if chroma_client:
    # 加载模型
    with st.spinner("正在加载嵌入模型..."):
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    with st.spinner("正在加载生成模型..."):
        generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

    models_loaded = embedding_model and generation_model and tokenizer

    if models_loaded:
        # 设置集合并索引数据（如果需要）
        with st.spinner("正在准备向量数据库..."):
            collection_ready = setup_chroma_collection(chroma_client, embedding_model)

        if collection_ready:
            st.success("✅ 系统准备就绪")

            # --- RAG 交互界面 ---
            st.subheader("💬 医疗问答")

            # 示例问题快速选择
            example_questions = [
                "感冒有什么症状？",
                "高血压患者应该注意什么？",
                "如何预防糖尿病？",
                "心脏病的早期症状有哪些？"
            ]

            cols = st.columns(4)
            selected_query = None
            for i, question in enumerate(example_questions):
                with cols[i]:
                    if st.button(question[:15] + "...", help=question, use_container_width=True):
                        selected_query = question

            query = st.text_input(
                "或输入您自己的问题：",
                value=selected_query if selected_query else "",
                placeholder="例如：感冒有什么症状？如何治疗？",
                key="query_input"
            )

            search_col1, search_col2, search_col3 = st.columns([1, 1, 8])
            with search_col1:
                search_btn = st.button("🔍 搜索答案", type="primary", use_container_width=True)
            with search_col2:
                if st.button("🔄 清空", use_container_width=True):
                    st.rerun()

            if search_btn and query:
                start_time = time.time()

                # 1. 检索相关文档
                with st.spinner("正在从知识库中检索相关信息..."):
                    retrieved_ids, distances, retrieved_docs = search_similar_documents(
                        chroma_client, query, embedding_model
                    )

                if not retrieved_docs:
                    st.warning("⚠️ 未找到相关医学资料。请尝试其他提问方式。")
                else:
                    # 2. 显示检索结果
                    st.subheader("📄 检索结果")

                    # 创建结果容器
                    result_container = st.container()

                    for i, doc in enumerate(retrieved_docs):
                        relevance = 1 - distances[i] if i < len(distances) else 0
                        relevance_color = "#10b981" if relevance > 0.5 else "#f59e0b" if relevance > 0.3 else "#ef4444"

                        with st.expander(f"📖 {i + 1}. {doc['title'][:50]}... (相关度: {relevance:.2f})",
                                         expanded=(i == 0)):
                            # 显示相关度条
                            st.markdown(f"""
                            <div style="margin: 5px 0; background: #f0f0f0; border-radius: 4px; height: 6px;">
                                <div style="background: {relevance_color}; width: {relevance * 100}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.caption(f"📎 来源: {doc.get('source_file', '未知')}")
                            st.markdown("**内容摘要:**")
                            st.info(doc['abstract'][:400] + ("..." if len(doc['abstract']) > 400 else ""))

                    st.divider()

                    # 3. 生成答案
                    st.subheader("🤖 AI 生成的回答")
                    with st.spinner("正在综合检索内容生成回答..."):
                        answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)

                    # 美化答案显示
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2px;
                        border-radius: 10px;
                        margin: 15px 0;
                    ">
                    <div style="
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                    ">
                    """, unsafe_allow_html=True)

                    st.markdown(answer)

                    st.markdown("</div></div>", unsafe_allow_html=True)

                    # 显示性能信息
                    end_time = time.time()
                    elapsed = end_time - start_time

                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.metric("⏱️ 耗时", f"{elapsed:.2f}s")
                    with info_col2:
                        st.metric("📄 文档数", len(retrieved_docs))
                    with info_col3:
                        avg_relevance = sum([1 - d for d in distances]) / len(distances) if distances else 0
                        st.metric("🎯 平均相关度", f"{avg_relevance:.3f}")
            elif search_btn and not query:
                st.warning("请输入问题后再搜索。")

            # --- 系统信息侧边栏 ---
            with st.sidebar:
                st.header("⚙️ 系统配置")

                # 使用小卡片显示配置
                config_info = f"""
                **向量数据库**: ChromaDB  
                **集合名称**: `{COLLECTION_NAME}`  
                **嵌入模型**: `{EMBEDDING_MODEL_NAME}`  
                **生成模型**: `{GENERATION_MODEL_NAME}`  
                **最大索引数**: `{MAX_ARTICLES_TO_INDEX}`  
                **检索数量**: `{TOP_K}`
                """
                st.markdown(config_info)

                st.divider()

                # 数据管理部分
                st.subheader("📊 数据管理")

                if st.button("🔄 重新处理数据", use_container_width=True):
                    st.info("请在命令行中运行: python data_manager.py")

                if st.button("📋 查看数据统计", use_container_width=True):
                    if id_to_doc_map:
                        st.metric("知识库文档数", len(id_to_doc_map))

                        # 显示文档统计
                        st.markdown("**最近添加的文档:**")
                        for i, (doc_id, doc) in enumerate(list(id_to_doc_map.items())[:3]):
                            st.caption(f"• {doc['title'][:30]}...")
                    else:
                        st.warning("文档映射为空")

                st.divider()

                # 系统信息
                st.subheader("ℹ️ 系统信息")
                st.caption("版本: 1.0.0")
                st.caption("最后更新: 2026-1-6")
                st.caption("开发者: 陈轩浩")

                st.markdown("---")
                st.markdown("⚠️ **免责声明**: 本系统提供的信息仅供参考，不能替代专业医疗建议。")

        else:
            st.error("向量数据库初始化失败，请检查数据文件。")
    else:
        st.error("模型加载失败，请检查配置和网络连接。")
else:
    st.error("ChromaDB 客户端初始化失败。")