"""
EmbeddingLab - Embedding 工具函数
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from functools import lru_cache

# Word2Vec 预训练模型的词汇表（精简版，用于演示）
DEMO_WORD_VECTORS = None

# 模型缓存
_word2vec_model = None
_sentence_transformer_cache = {}


def load_word2vec_model():
    """加载 Word2Vec 模型（使用 gensim）"""
    global _word2vec_model
    if _word2vec_model is not None:
        return _word2vec_model
    
    try:
        import gensim.downloader as api
        _word2vec_model = api.load("glove-wiki-gigaword-100")  # 100维的GloVe模型，较小
        return _word2vec_model
    except Exception as e:
        print(f"加载 Word2Vec 模型失败: {e}")
        return None


def load_sentence_transformer(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """加载 Sentence Transformer 模型"""
    global _sentence_transformer_cache
    
    if model_name in _sentence_transformer_cache:
        return _sentence_transformer_cache[model_name]
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _sentence_transformer_cache[model_name] = model
        return model
    except Exception as e:
        print(f"加载 Sentence Transformer 失败: {e}")
        return None


def get_word_vector(model, word: str) -> Optional[np.ndarray]:
    """获取单词的向量表示"""
    try:
        return model[word]
    except KeyError:
        return None


def vector_arithmetic(model, positive: List[str], negative: List[str], topn: int = 10) -> List[Tuple[str, float]]:
    """
    执行向量运算：positive - negative
    例如：King - Man + Woman = ? (positive=['king', 'woman'], negative=['man'])
    """
    try:
        result = model.most_similar(positive=positive, negative=negative, topn=topn)
        return result
    except Exception as e:
        return []


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_similarity_matrix(vectors1: List[np.ndarray], vectors2: List[np.ndarray]) -> np.ndarray:
    """计算两组向量之间的相似度矩阵"""
    matrix = np.zeros((len(vectors1), len(vectors2)))
    for i, v1 in enumerate(vectors1):
        for j, v2 in enumerate(vectors2):
            matrix[i, j] = cosine_similarity(v1, v2)
    return matrix


class DimensionReductionError(Exception):
    """降维算法的自定义错误，包含用户友好的提示信息"""
    pass


# 各算法的最小样本数要求
MIN_SAMPLES = {
    "pca": 2,
    "tsne": 6,   # t-SNE 建议 perplexity >= 5，所以需要至少 6 个样本
    "umap": 15,  # UMAP 默认 n_neighbors=15，需要至少 15 个样本才能稳定工作
}


def reduce_dimensions(vectors: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
    """
    降维处理
    
    Args:
        vectors: 输入向量矩阵
        method: 降维方法 (pca, tsne, umap)
        n_components: 目标维度
        
    Returns:
        降维后的坐标
        
    Raises:
        DimensionReductionError: 当样本数不满足算法要求时
    """
    n_samples = len(vectors)
    
    if n_samples == 0:
        raise DimensionReductionError("没有数据点可以可视化")
    
    min_required = MIN_SAMPLES.get(method, 2)
    
    if method == "pca":
        if n_samples < 2:
            raise DimensionReductionError(
                f"PCA 至少需要 2 个数据点，当前只有 {n_samples} 个"
            )
        from sklearn.decomposition import PCA
        n_components = min(n_components, n_samples, vectors.shape[1] if len(vectors.shape) > 1 else 1)
        reducer = PCA(n_components=n_components)
        
    elif method == "tsne":
        if n_samples < 6:
            raise DimensionReductionError(
                f"t-SNE 至少需要 6 个数据点（perplexity 参数要求），当前只有 {n_samples} 个，建议添加更多数据点或使用 PCA 算法"
            )
        from sklearn.manifold import TSNE
        # perplexity 必须 < n_samples
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        
    elif method == "umap":
        if n_samples < 15:
            raise DimensionReductionError(
                f"UMAP 至少需要 15 个数据点（n_neighbors 参数要求），当前只有 {n_samples} 个，建议添加更多数据点或使用 PCA/t-SNE 算法"
            )
        import umap
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        
    else:
        raise ValueError(f"未知的降维方法: {method}")
    
    return reducer.fit_transform(vectors)


# ================================
# Embedding 模型配置
# ================================

EMBEDDING_MODELS = {
    "Dense Models": {
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "name": "Multilingual MiniLM",
            "type": "dense",
            "dim": 384,
            "description": "多语言轻量级模型"
        },
        "all-MiniLM-L6-v2": {
            "name": "MiniLM-L6",
            "type": "dense",
            "dim": 384,
            "description": "英文轻量级模型"
        },
        "BAAI/bge-small-zh-v1.5": {
            "name": "BGE-Small-ZH",
            "type": "dense",
            "dim": 512,
            "description": "中文小模型"
        },
    },
    "Classical Models": {
        "tfidf": {
            "name": "TF-IDF",
            "type": "sparse",
            "dim": "variable",
            "description": "词频-逆文档频率"
        },
        "bm25": {
            "name": "BM25",
            "type": "sparse",
            "dim": "variable",
            "description": "概率检索模型"
        },
    }
}


def get_text_embedding(text: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> Optional[np.ndarray]:
    """获取文本的 embedding"""
    if model_name in ["tfidf", "bm25"]:
        return None  # 这些需要整个语料库
    
    model = load_sentence_transformer(model_name)
    if model is None:
        return None
    
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def get_batch_embeddings(texts: List[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> Optional[np.ndarray]:
    """批量获取文本的 embeddings"""
    if model_name in ["tfidf", "bm25"]:
        return compute_sparse_embeddings(texts, model_name)
    
    model = load_sentence_transformer(model_name)
    if model is None:
        return None
    
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def chinese_tokenizer(text: str) -> List[str]:
    """中文分词器，支持中英文混合"""
    import jieba
    import re
    
    # 使用 jieba 分词
    tokens = jieba.lcut(text)
    
    # 过滤掉空白和标点符号
    tokens = [t.strip() for t in tokens if t.strip() and not re.match(r'^[\s\W]+$', t)]
    
    return tokens


def compute_sparse_embeddings(texts: List[str], method: str = "tfidf") -> np.ndarray:
    """计算稀疏向量（TF-IDF 或 BM25），支持中文分词"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 检测是否包含中文
    import re
    has_chinese = any(re.search(r'[\u4e00-\u9fff]', text) for text in texts)
    
    # 如果包含中文，使用 jieba 分词
    tokenizer = chinese_tokenizer if has_chinese else None
    
    if method == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None if tokenizer else r'(?u)\b\w+\b')
        vectors = vectorizer.fit_transform(texts).toarray()
        return vectors
    elif method == "bm25":
        # 简化的 BM25 实现，使用 TF-IDF 的变体
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer, 
            token_pattern=None if tokenizer else r'(?u)\b\w+\b',
            use_idf=True, 
            norm='l2', 
            sublinear_tf=True
        )
        vectors = vectorizer.fit_transform(texts).toarray()
        return vectors
    
    return np.array([])


def compute_anisotropy(vectors: np.ndarray, sample_size: int = 100) -> Tuple[float, float]:
    """
    计算向量空间的各向异性
    返回：(平均余弦相似度, 标准差)
    """
    n = len(vectors)
    if n < 2:
        return 0.0, 0.0
    
    # 随机采样
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        sampled = vectors[indices]
    else:
        sampled = vectors
    
    similarities = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            sim = cosine_similarity(sampled[i], sampled[j])
            similarities.append(sim)
    
    if not similarities:
        return 0.0, 0.0
    
    return float(np.mean(similarities)), float(np.std(similarities))


def whitening_transform(vectors: np.ndarray) -> np.ndarray:
    """
    对向量进行白化处理，缓解各向异性问题
    """
    # 中心化
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    
    # 计算协方差矩阵
    cov = np.cov(centered.T)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 防止除零
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # 白化变换
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    whitened = centered @ whitening_matrix
    return whitened


# ================================
# 预置数据集
# ================================

PRESET_DATASETS = {
    "news_categories": {
        "name": "新闻分类",
        "description": "不同类别的新闻标题",
        "data": [
            {"text": "股市今日大涨，上证指数突破3500点", "label": "财经"},
            {"text": "央行宣布降准0.5个百分点", "label": "财经"},
            {"text": "比特币价格创历史新高", "label": "财经"},
            {"text": "新能源汽车销量同比增长50%", "label": "财经"},
            {"text": "国足世预赛2-0战胜对手", "label": "体育"},
            {"text": "NBA季后赛火热进行中", "label": "体育"},
            {"text": "奥运会开幕式精彩纷呈", "label": "体育"},
            {"text": "网球公开赛决赛今晚上演", "label": "体育"},
            {"text": "新款iPhone正式发布", "label": "科技"},
            {"text": "人工智能突破语言理解障碍", "label": "科技"},
            {"text": "量子计算机取得重大进展", "label": "科技"},
            {"text": "SpaceX成功发射星链卫星", "label": "科技"},
            {"text": "知名演员新电影上映", "label": "娱乐"},
            {"text": "热门综艺节目收视率创新高", "label": "娱乐"},
            {"text": "流行歌手发布新专辑", "label": "娱乐"},
            {"text": "电影节颁奖典礼举行", "label": "娱乐"},
        ]
    },
    "sentiment": {
        "name": "情感分析",
        "description": "正面/负面情感文本",
        "data": [
            {"text": "这个产品太棒了，强烈推荐！", "label": "正面"},
            {"text": "服务非常周到，很满意", "label": "正面"},
            {"text": "质量很好，物超所值", "label": "正面"},
            {"text": "体验很棒，下次还会来", "label": "正面"},
            {"text": "太失望了，完全不值这个价", "label": "负面"},
            {"text": "服务态度太差，不会再来", "label": "负面"},
            {"text": "质量很差，浪费钱", "label": "负面"},
            {"text": "等了很久，体验很糟糕", "label": "负面"},
            {"text": "还行吧，一般般", "label": "中性"},
            {"text": "没什么特别的感觉", "label": "中性"},
            {"text": "马马虎虎，凑合用", "label": "中性"},
            {"text": "普普通通，不好不坏", "label": "中性"},
        ]
    },
    "multilingual": {
        "name": "中英混合",
        "description": "中英文语义对应",
        "data": [
            {"text": "我爱你", "label": "中文"},
            {"text": "I love you", "label": "英文"},
            {"text": "今天天气很好", "label": "中文"},
            {"text": "The weather is nice today", "label": "英文"},
            {"text": "人工智能改变世界", "label": "中文"},
            {"text": "Artificial intelligence changes the world", "label": "英文"},
            {"text": "学习是终身的事业", "label": "中文"},
            {"text": "Learning is a lifelong journey", "label": "英文"},
            {"text": "时间就是金钱", "label": "中文"},
            {"text": "Time is money", "label": "英文"},
            {"text": "知识就是力量", "label": "中文"},
            {"text": "Knowledge is power", "label": "英文"},
        ]
    }
}


# ================================
# 颜色配置
# ================================

LABEL_COLORS = {
    # 新闻分类
    "财经": "#2563EB",
    "体育": "#059669",
    "科技": "#7C3AED",
    "娱乐": "#DC2626",
    # 情感分析
    "正面": "#059669",
    "负面": "#DC2626",
    "中性": "#6B7280",
    # 多语言
    "中文": "#2563EB",
    "英文": "#D97706",
}


def get_label_color(label: str) -> str:
    """获取标签对应的颜色"""
    return LABEL_COLORS.get(label, "#6B7280")
