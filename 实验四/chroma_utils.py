# chroma_utils.py（简化稳定版）
import streamlit as st
import chromadb
from chromadb.config import Settings
import hashlib
import os

from config import (
    COLLECTION_NAME, EMBEDDING_DIM, EMBEDDING_MODEL_NAME,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map
)


@st.cache_resource
def get_chroma_client():
    """Initializes and returns a ChromaDB persistent client."""
    try:
        # 数据将持久化在 ./chroma_data 目录下
        client = chromadb.PersistentClient(path="./chroma_data")
        st.success("ChromaDB 客户端初始化成功！")
        return client
    except Exception as e:
        st.error(f"初始化 ChromaDB 客户端失败: {e}")
        return None


def setup_chroma_collection(_client, embedding_model):
    """
    设置或获取ChromaDB集合。
    返回布尔值表示是否成功。
    """
    global id_to_doc_map

    if not _client:
        st.error("ChromaDB 客户端不可用。")
        return False

    try:
        # 获取或创建集合
        collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        st.write(f"ChromaDB 集合 '{COLLECTION_NAME}' 已就绪。")

        # 检查当前集合中的文档数量
        current_count = collection.count()
        st.write(f"集合中现有 {current_count} 条向量。")

        # 如果集合为空，加载数据
        if current_count == 0:
            from data_utils import load_data
            data = load_data("./data/processed_data.json")
            data_to_index = data[:MAX_ARTICLES_TO_INDEX]

            docs_to_embed = []
            metadatas = []
            ids = []

            for doc in data_to_index:
                content = doc.get('abstract', '')
                title = doc.get('title', '无标题')

                if not content:
                    continue

                # 创建ID
                doc_id = hashlib.md5(f"{title}_{content}".encode()).hexdigest()[:20]

                # 存储到映射
                id_to_doc_map[doc_id] = {
                    'title': title,
                    'abstract': content,
                    'content': f"标题：{title}\n内容：{content}",
                    'source_file': doc.get('source_file', ''),
                    'chunk_index': doc.get('chunk_index', 0)
                }

                # 准备数据
                docs_to_embed.append(content)
                metadatas.append({
                    "source_file": doc.get('source_file', ''),
                    "title": title,
                    "chunk_index": doc.get('chunk_index', 0)
                })
                ids.append(doc_id)

            if docs_to_embed:
                st.warning(f"开始嵌入并索引 {len(docs_to_embed)} 个文档...")
                embeddings = embedding_model.encode(docs_to_embed).tolist()

                collection.add(
                    embeddings=embeddings,
                    documents=docs_to_embed,
                    metadatas=metadatas,
                    ids=ids
                )
                st.success(f"成功索引 {len(ids)} 个文档。")

        return True

    except Exception as e:
        st.error(f"设置 ChromaDB 集合时出错: {e}")
        return False


def search_similar_documents(_client, query, embedding_model):
    """Searches ChromaDB for documents similar to the query."""
    if not _client or not embedding_model:
        st.error("ChromaDB 客户端或嵌入模型不可用。")
        return [], [], []

    try:
        collection = _client.get_collection(name=COLLECTION_NAME)

        # 生成查询向量
        query_embedding = embedding_model.encode([query]).tolist()

        # 执行查询
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K
        )

        # 解析结果
        if results and results['ids'][0]:
            retrieved_ids = results['ids'][0]
            distances = results['distances'][0] if results['distances'] else []

            # 构建文档列表
            retrieved_docs = []
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in id_to_doc_map:
                    retrieved_docs.append(id_to_doc_map[doc_id])
                else:
                    # 如果映射中没有，使用从ChromaDB返回的信息
                    doc_content = results['documents'][0][i] if i < len(results['documents'][0]) else ""
                    metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                    retrieved_docs.append({
                        'title': metadata.get('title', '未知标题'),
                        'abstract': doc_content,
                        'content': doc_content,
                        'source_file': metadata.get('source_file', ''),
                        'chunk_index': metadata.get('chunk_index', 0)
                    })

            return retrieved_ids, distances, retrieved_docs
        else:
            return [], [], []

    except Exception as e:
        st.error(f"ChromaDB 查询过程中出错: {e}")
        return [], [], []