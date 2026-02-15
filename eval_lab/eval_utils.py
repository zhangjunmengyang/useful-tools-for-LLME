"""
评测工具函数 - 各种评测指标的计算实现

包含 BLEU, ROUGE, Exact Match, F1 Score 等指标的简化实现
由于项目不依赖外部评测库，这里提供纯 Python 的实现
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import math


def tokenize_simple(text: str) -> List[str]:
    """
    简单的分词函数，用于 BLEU/ROUGE 计算
    
    Args:
        text: 输入文本
    
    Returns:
        分词后的 token 列表
    """
    # 移除标点和多余空格，转小写
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = text.split()
    return tokens


def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    计算 Exact Match 准确率
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
    
    Returns:
        EM 分数 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果和参考答案数量不匹配")
    
    matches = 0
    for pred, ref in zip(predictions, references):
        if pred.strip().lower() == ref.strip().lower():
            matches += 1
    
    return matches / len(predictions)


def f1_score(predictions: List[str], references: List[str]) -> float:
    """
    计算 F1 分数（基于 token 级别）
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
    
    Returns:
        平均 F1 分数
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果和参考答案数量不匹配")
    
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize_simple(pred))
        ref_tokens = set(tokenize_simple(ref))
        
        if not pred_tokens and not ref_tokens:
            f1_scores.append(1.0)
            continue
        elif not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        
        intersection = pred_tokens.intersection(ref_tokens)
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    
    return np.mean(f1_scores)


def bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
    """
    计算 BLEU 分数的简化版本（单参考答案）
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
        n: N-gram 最大长度
    
    Returns:
        BLEU 分数
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果和参考答案数量不匹配")
    
    if not predictions:
        return 0.0
    
    # 计算所有样本的 n-gram 精度
    all_precisions = {i: [] for i in range(1, n + 1)}
    brevity_penalties = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_simple(pred)
        ref_tokens = tokenize_simple(ref)
        
        # 简化的 brevity penalty
        if len(pred_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
        else:
            bp = 1.0
        brevity_penalties.append(bp)
        
        # 计算 n-gram 精度
        for i in range(1, n + 1):
            if len(pred_tokens) < i:
                all_precisions[i].append(0.0)
                continue
            
            # 生成 n-gram
            pred_ngrams = Counter()
            ref_ngrams = Counter()
            
            for j in range(len(pred_tokens) - i + 1):
                ngram = tuple(pred_tokens[j:j + i])
                pred_ngrams[ngram] += 1
            
            for j in range(len(ref_tokens) - i + 1):
                ngram = tuple(ref_tokens[j:j + i])
                ref_ngrams[ngram] += 1
            
            # 计算精度
            matches = 0
            total_pred = sum(pred_ngrams.values())
            
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            precision = matches / max(total_pred, 1)
            all_precisions[i].append(precision)
    
    # 计算几何平均
    avg_precisions = []
    for i in range(1, n + 1):
        avg_p = np.mean(all_precisions[i]) if all_precisions[i] else 0.0
        avg_precisions.append(avg_p + 1e-10)  # 避免对数为负无穷
    
    # BLEU = brevity_penalty * exp(avg(log(precisions)))
    log_sum = sum(math.log(p) for p in avg_precisions)
    geometric_mean = math.exp(log_sum / n)
    avg_bp = np.mean(brevity_penalties)
    
    return avg_bp * geometric_mean


def rouge_l_score(predictions: List[str], references: List[str]) -> float:
    """
    计算 ROUGE-L 分数（基于最长公共子序列）
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
    
    Returns:
        ROUGE-L F1 分数
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果和参考答案数量不匹配")
    
    def lcs_length(x: List[str], y: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        
        # 动态规划计算 LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_simple(pred)
        ref_tokens = tokenize_simple(ref)
        
        if not pred_tokens and not ref_tokens:
            rouge_scores.append(1.0)
            continue
        elif not pred_tokens or not ref_tokens:
            rouge_scores.append(0.0)
            continue
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        
        if precision + recall == 0:
            rouge_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            rouge_scores.append(f1)
    
    return np.mean(rouge_scores)


def bertscore_mock(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    BERTScore 的 mock 实现（用简单的语义相似度模拟）
    实际的 BERTScore 需要 BERT embeddings，这里用基于词汇重叠的简化版本
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
    
    Returns:
        包含 precision, recall, f1 的字典
    """
    if len(predictions) != len(references):
        raise ValueError("预测结果和参考答案数量不匹配")
    
    precisions, recalls, f1s = [], [], []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize_simple(pred))
        ref_tokens = set(tokenize_simple(ref))
        
        if not pred_tokens and not ref_tokens:
            precisions.append(1.0)
            recalls.append(1.0) 
            f1s.append(1.0)
            continue
        elif not pred_tokens or not ref_tokens:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        
        intersection = pred_tokens.intersection(ref_tokens)
        
        # 模拟语义相似度（实际 BERTScore 会计算 embedding 相似度）
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)
        
        # 添加一些随机性以模拟语义理解
        semantic_bonus = min(0.3, len(intersection) * 0.1)
        precision = min(1.0, precision + semantic_bonus)
        recall = min(1.0, recall + semantic_bonus)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }


def calculate_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    计算所有评测指标
    
    Args:
        predictions: 预测结果列表
        references: 参考答案列表
    
    Returns:
        包含所有指标的字典
    """
    if not predictions or not references:
        return {metric: 0.0 for metric in ['exact_match', 'f1', 'bleu', 'rouge_l', 'bertscore_f1']}
    
    try:
        metrics = {}
        
        # 基础指标
        metrics['exact_match'] = exact_match(predictions, references)
        metrics['f1'] = f1_score(predictions, references)
        metrics['bleu'] = bleu_score(predictions, references)
        metrics['rouge_l'] = rouge_l_score(predictions, references)
        
        # BERTScore
        bert_scores = bertscore_mock(predictions, references)
        metrics['bertscore_f1'] = bert_scores['f1']
        
        return metrics
    
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {metric: 0.0 for metric in ['exact_match', 'f1', 'bleu', 'rouge_l', 'bertscore_f1']}


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    格式化指标结果为表格
    
    Args:
        metrics: 指标字典
    
    Returns:
        Markdown 格式的表格
    """
    metric_names = {
        'exact_match': 'Exact Match',
        'f1': 'F1 Score', 
        'bleu': 'BLEU',
        'rouge_l': 'ROUGE-L',
        'bertscore_f1': 'BERTScore F1'
    }
    
    table_rows = []
    table_rows.append("| 指标 | 分数 |")
    table_rows.append("|------|------|")
    
    for key, value in metrics.items():
        if key in metric_names:
            name = metric_names[key]
            score = f"{value:.4f}"
            table_rows.append(f"| {name} | {score} |")
    
    return "\n".join(table_rows)


def generate_eval_report(dataset_name: str, model_name: str, metrics: Dict[str, float], 
                        sample_predictions: List[Tuple[str, str, str]] = None) -> str:
    """
    生成 Markdown 格式的评测报告
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称  
        metrics: 评测指标结果
        sample_predictions: 样本预测结果 (question, reference, prediction)
    
    Returns:
        Markdown 格式的报告
    """
    report_lines = []
    
    # 报告头部
    report_lines.append(f"# 模型评测报告")
    report_lines.append("")
    report_lines.append(f"**模型**: {model_name}")
    report_lines.append(f"**数据集**: {dataset_name}")
    report_lines.append(f"**评测时间**: {'-'}")  # 简化，不添加真实时间
    report_lines.append("")
    
    # 整体指标
    report_lines.append("## 整体指标")
    report_lines.append("")
    report_lines.append(format_metrics_table(metrics))
    report_lines.append("")
    
    # 指标解释
    report_lines.append("## 指标说明")
    report_lines.append("")
    report_lines.append("- **Exact Match**: 预测结果与参考答案完全匹配的比例")
    report_lines.append("- **F1 Score**: 基于 token 级别的 F1 分数")
    report_lines.append("- **BLEU**: 机器翻译质量评价指标，基于 n-gram 匹配")
    report_lines.append("- **ROUGE-L**: 基于最长公共子序列的文本相似度")
    report_lines.append("- **BERTScore F1**: 基于语义理解的相似度评分（mock版本）")
    report_lines.append("")
    
    # 样本展示
    if sample_predictions:
        report_lines.append("## 预测样本")
        report_lines.append("")
        
        for i, (question, reference, prediction) in enumerate(sample_predictions[:3]):
            report_lines.append(f"### 样本 {i + 1}")
            report_lines.append("")
            report_lines.append(f"**问题**: {question}")
            report_lines.append("")
            report_lines.append(f"**参考答案**: {reference}")
            report_lines.append("")  
            report_lines.append(f"**模型预测**: {prediction}")
            report_lines.append("")
    
    # 评测建议  
    report_lines.append("## 评测建议")
    report_lines.append("")
    
    if metrics.get('exact_match', 0) < 0.3:
        report_lines.append("- ⚠️ Exact Match 分数较低，建议检查输出格式")
    
    if metrics.get('f1', 0) > 0.8:
        report_lines.append("- ✅ F1 分数表现优秀，模型语义理解能力较强")
    elif metrics.get('f1', 0) < 0.5:
        report_lines.append("- ⚠️ F1 分数较低，建议优化模型或数据")
    
    if metrics.get('bleu', 0) > 0.4:
        report_lines.append("- ✅ BLEU 分数优秀，适合生成任务")
    
    return "\n".join(report_lines)