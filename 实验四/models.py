# models.py（修复版，强制本地加载）
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


@st.cache_resource
def load_generation_model(model_name):
    """Loads the Hugging Face generative model and tokenizer."""
    st.write(f"Loading generation model: {model_name}...")
    try:
        # 设置环境变量强制离线（关键修复）
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'

        # 使用绝对路径，确保从本地加载
        cache_dir = 'D:/.cache/huggingface'

        # 首先检查本地是否有模型文件
        model_path = os.path.join(cache_dir, 'models--Qwen--Qwen2.5-0.5B')

        if os.path.exists(model_path):
            st.write("✅ 检测到本地模型缓存，尝试从本地加载...")

            # 尝试从snapshots目录加载
            snapshots_path = os.path.join(model_path, 'snapshots')
            if os.path.exists(snapshots_path):
                # 获取最新的snapshot
                snapshot_dirs = [d for d in os.listdir(snapshots_path)
                                 if os.path.isdir(os.path.join(snapshots_path, d))]

                if snapshot_dirs:
                    latest_snapshot = max(snapshot_dirs)  # 通常按时间排序
                    local_model_path = os.path.join(snapshots_path, latest_snapshot)

                    tokenizer = AutoTokenizer.from_pretrained(
                        local_model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        device_map="auto",
                        torch_dtype=torch.float32  # 使用float32避免CUDA错误
                    )

                    st.success("✅ 从本地缓存加载生成模型成功！")
                    return model, tokenizer

        # 如果上述方法失败，尝试标准的本地加载方式
        st.write("⚠️ 使用标准本地加载方式...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=True  # 关键：强制本地加载
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=True,  # 关键：强制本地加载
            device_map="auto",
            torch_dtype=torch.float32  # 使用float32避免CUDA错误
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        st.success("Generation model and tokenizer loaded.")
        return model, tokenizer

    except Exception as e:
        st.error(f"Failed to load generation model: {e}")

        # 更详细的错误信息
        import traceback
        error_details = traceback.format_exc()
        st.code(f"详细错误信息:\n{error_details}")

        # 检查缓存目录是否存在
        cache_dir = 'D:/.cache/huggingface'
        if not os.path.exists(cache_dir):
            st.error(f"缓存目录不存在: {cache_dir}")
        else:
            st.error(f"缓存目录存在，但无法加载模型。请检查是否有 Qwen/Qwen2.5-0.5B 的模型文件")

        return None, None