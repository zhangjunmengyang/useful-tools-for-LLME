"""
RAG 工具函数

提供文本分块和检索相关的核心功能
"""

from typing import List, Tuple, Dict
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np


# 缓存加载的 embedding 模型
_embedding_model_cache = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    获取 embedding 模型（带缓存）

    Args:
        model_name: 模型名称

    Returns:
        SentenceTransformer 模型实例
    """
    if model_name not in _embedding_model_cache:
        try:
            _embedding_model_cache[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    return _embedding_model_cache[model_name]


def chunk_fixed_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    固定大小分块策略

    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        overlap: 重叠大小（字符数）

    Returns:
        分块列表
    """
    if chunk_size <= 0:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk.strip():  # 只添加非空块
            chunks.append(chunk)

        # 如果已到达文本末尾，退出循环
        if end >= text_len:
            break

        # 计算下一个起始位置
        start = start + chunk_size - overlap

        # 防止无限循环（当 overlap >= chunk_size 时）
        if overlap >= chunk_size and start <= chunk_size:
            start = chunk_size

    return chunks


def chunk_recursive(text: str, chunk_size: int, overlap: int, separators: List[str] = None) -> List[str]:
    """
    递归分块策略（优先按段落、句子分割）

    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        overlap: 重叠大小（字符数）
        separators: 分隔符列表，优先级从高到低

    Returns:
        分块列表
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if chunk_size <= 0:
        return []

    def _split_text(text: str, seps: List[str]) -> List[str]:
        """递归分割文本"""
        if not seps or len(text) <= chunk_size:
            return [text] if text.strip() else []

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep == "":
            # 最后一级：固定大小切分
            return chunk_fixed_size(text, chunk_size, overlap)

        # 按当前分隔符切分
        parts = text.split(sep)
        chunks = []
        current_chunk = ""

        for part in parts:
            if len(current_chunk) + len(sep) + len(part) <= chunk_size:
                # 可以合并到当前块
                if current_chunk:
                    current_chunk += sep + part
                else:
                    current_chunk = part
            else:
                # 当前块已满
                if current_chunk:
                    chunks.append(current_chunk)

                # 如果单个 part 过大，递归处理
                if len(part) > chunk_size:
                    sub_chunks = _split_text(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    return _split_text(text, separators)


def chunk_by_sentence(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    按句子分块（基于句号、问号、感叹号）

    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        overlap: 重叠大小（字符数）

    Returns:
        分块列表
    """
    # 使用正则表达式分割句子
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    if chunk_size <= 0:
        return []

    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 如果当前块加上新句子不超过限制
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk)

                # 准备 overlap buffer（取当前块的最后部分）
                if overlap > 0:
                    overlap_buffer = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                else:
                    overlap_buffer = ""

            # 开始新块（带 overlap）
            if overlap_buffer:
                current_chunk = overlap_buffer + " " + sentence
            else:
                current_chunk = sentence

    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def compute_chunk_stats(chunks: List[str]) -> Dict[str, any]:
    """
    计算分块统计信息

    Args:
        chunks: 分块列表

    Returns:
        统计信息字典
    """
    if not chunks:
        return {
            "num_chunks": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_chars": 0
        }

    lengths = [len(chunk) for chunk in chunks]

    return {
        "num_chunks": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths)
    }


def compute_similarity(query: str, documents: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[Tuple[int, float]]:
    """
    计算 query 与文档的余弦相似度

    Args:
        query: 查询文本
        documents: 文档列表
        model_name: embedding 模型名称

    Returns:
        (文档索引, 相似度分数) 列表，按相似度降序排列
    """
    if not documents:
        return []

    try:
        model = get_embedding_model(model_name)

        # 编码 query 和文档
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        doc_embeddings = model.encode(documents, convert_to_numpy=True)

        # 计算余弦相似度
        similarities = []
        for idx, doc_emb in enumerate(doc_embeddings):
            # 余弦相似度 = dot(A, B) / (||A|| * ||B||)
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append((idx, float(similarity)))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    except Exception as e:
        raise RuntimeError(f"Similarity computation failed: {str(e)}")


# 默认的示例文本
DEFAULT_CHUNKING_TEXT = """Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing.

The core idea behind RAG is to enhance language models with external knowledge retrieved from a large corpus of documents. When a user asks a question, the system first retrieves relevant documents from a knowledge base, then uses these documents as context to generate a more informed and accurate response.

RAG systems typically consist of three main components: a retriever, a knowledge base, and a generator. The retriever finds relevant documents based on semantic similarity to the input query. The knowledge base stores the documents that can be retrieved. The generator, usually a large language model, produces the final output by conditioning on both the input query and the retrieved documents.

One of the key advantages of RAG is that it allows language models to access up-to-date information without requiring expensive retraining. The knowledge base can be updated independently of the model, making it easier to incorporate new information and correct outdated facts.

Text chunking is a critical preprocessing step in RAG systems. It involves breaking down large documents into smaller, manageable pieces that can be efficiently retrieved and processed. Different chunking strategies can significantly impact the quality of retrieval and generation."""


DEFAULT_RETRIEVAL_DOCS = """Document 1: Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

Document 2: Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. Common algorithms include decision trees, neural networks, and support vector machines.

Document 3: Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like sentiment analysis, named entity recognition, and machine translation.

Document 4: Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. It has achieved remarkable success in computer vision, speech recognition, and language understanding.

Document 5: The transformer architecture, introduced in 2017, revolutionized NLP by using self-attention mechanisms. It forms the basis of models like BERT, GPT, and T5."""


DEFAULT_RETRIEVAL_QUERY = "What is natural language processing?"
