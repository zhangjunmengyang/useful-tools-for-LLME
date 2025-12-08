"""
Dataset 透视镜 - HuggingFace Dataset 流式预览与分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import json


def analyze_field_content(values: List[Any], field_name: str) -> Dict:
    """分析字段内容的统计信息"""
    stats = {
        'name': field_name,
        'total': len(values),
        'null_count': 0,
        'unique_count': 0,
        'avg_length': 0,
        'min_length': 0,
        'max_length': 0,
        'type': 'unknown'
    }
    
    # 过滤空值
    non_null = [v for v in values if v is not None and v != '']
    stats['null_count'] = len(values) - len(non_null)
    
    if not non_null:
        return stats
    
    # 判断类型
    first_val = non_null[0]
    if isinstance(first_val, str):
        stats['type'] = 'string'
        lengths = [len(str(v)) for v in non_null]
        stats['avg_length'] = np.mean(lengths)
        stats['min_length'] = np.min(lengths)
        stats['max_length'] = np.max(lengths)
        # 唯一值数量（限制计算量）
        if len(non_null) <= 1000:
            stats['unique_count'] = len(set(non_null))
    elif isinstance(first_val, (int, float)):
        stats['type'] = 'number'
        stats['avg_length'] = np.mean(non_null)
        stats['min_length'] = np.min(non_null)
        stats['max_length'] = np.max(non_null)
    elif isinstance(first_val, list):
        stats['type'] = 'list'
        lengths = [len(v) for v in non_null]
        stats['avg_length'] = np.mean(lengths)
        stats['min_length'] = np.min(lengths)
        stats['max_length'] = np.max(lengths)
    elif isinstance(first_val, dict):
        stats['type'] = 'dict'
        stats['unique_count'] = len(non_null)
    
    return stats


def render_length_distribution(df: pd.DataFrame, field: str) -> go.Figure:
    """渲染字段长度分布直方图"""
    lengths = df[field].astype(str).str.len()
    
    fig = go.Figure(data=go.Histogram(
        x=lengths,
        nbinsx=30,
        marker_color='#2563EB',
        opacity=0.8
    ))
    
    fig.add_vline(x=lengths.mean(), line_dash="dash", line_color="#DC2626",
                  annotation_text=f"平均: {lengths.mean():.0f}")
    fig.add_vline(x=lengths.median(), line_dash="dot", line_color="#059669",
                  annotation_text=f"中位数: {lengths.median():.0f}")
    
    fig.update_layout(
        title=f"'{field}' 长度分布",
        xaxis_title="字符长度",
        yaxis_title="样本数",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def render_word_count_distribution(df: pd.DataFrame, field: str) -> go.Figure:
    """渲染字段词数分布"""
    word_counts = df[field].astype(str).str.split().str.len()
    
    fig = go.Figure(data=go.Histogram(
        x=word_counts,
        nbinsx=30,
        marker_color='#7C3AED',
        opacity=0.8
    ))
    
    fig.add_vline(x=word_counts.mean(), line_dash="dash", line_color="#DC2626",
                  annotation_text=f"平均: {word_counts.mean():.0f}")
    
    fig.update_layout(
        title=f"'{field}' 词数分布 (空格分词)",
        xaxis_title="词数",
        yaxis_title="样本数",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def render_field_stats_chart(field_stats: List[Dict]) -> go.Figure:
    """渲染字段统计汇总图"""
    string_fields = [f for f in field_stats if f['type'] == 'string']
    
    if not string_fields:
        return None
    
    names = [f['name'] for f in string_fields]
    avg_lengths = [f['avg_length'] for f in string_fields]
    null_rates = [f['null_count'] / f['total'] * 100 for f in string_fields]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='平均长度',
        x=names,
        y=avg_lengths,
        marker_color='#2563EB',
        yaxis='y',
        offsetgroup=0
    ))
    
    fig.add_trace(go.Bar(
        name='空值率 (%)',
        x=names,
        y=null_rates,
        marker_color='#DC2626',
        yaxis='y2',
        offsetgroup=1
    ))
    
    fig.update_layout(
        title="字段统计汇总",
        xaxis_title="字段",
        yaxis=dict(title="平均长度 (字符)", side='left'),
        yaxis2=dict(title="空值率 (%)", side='right', overlaying='y', range=[0, 100]),
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def check_data_quality(df: pd.DataFrame) -> Dict:
    """检查数据质量"""
    quality = {
        'total_samples': len(df),
        'duplicate_count': df.duplicated().sum(),
        'fields': {}
    }
    
    for col in df.columns:
        col_quality = {
            'null_count': df[col].isna().sum() + (df[col].astype(str) == '').sum(),
            'null_rate': 0,
            'empty_string_count': (df[col].astype(str).str.strip() == '').sum()
        }
        col_quality['null_rate'] = col_quality['null_count'] / len(df) * 100
        
        # 对文本字段检查更多
        if df[col].dtype == 'object':
            col_quality['very_short'] = (df[col].astype(str).str.len() < 10).sum()
            col_quality['very_long'] = (df[col].astype(str).str.len() > 10000).sum()
        
        quality['fields'][col] = col_quality
    
    return quality


def render_quality_report(quality: Dict) -> None:
    """渲染数据质量报告"""
    st.markdown("### 📋 数据质量报告")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总样本数", quality['total_samples'])
    with col2:
        dup_rate = quality['duplicate_count'] / quality['total_samples'] * 100
        st.metric("重复样本", f"{quality['duplicate_count']} ({dup_rate:.1f}%)")
    with col3:
        # 计算整体健康度
        total_issues = quality['duplicate_count']
        for field_quality in quality['fields'].values():
            total_issues += field_quality['null_count']
        health_score = max(0, 100 - (total_issues / quality['total_samples'] * 10))
        st.metric("健康度评分", f"{health_score:.0f}/100")
    
    # 字段质量详情
    st.markdown("#### 字段质量详情")
    
    quality_data = []
    for field, field_quality in quality['fields'].items():
        row = {
            '字段': field,
            '空值数': field_quality['null_count'],
            '空值率': f"{field_quality['null_rate']:.1f}%",
            '空字符串': field_quality['empty_string_count']
        }
        if 'very_short' in field_quality:
            row['过短 (<10字符)'] = field_quality['very_short']
        if 'very_long' in field_quality:
            row['过长 (>10k字符)'] = field_quality['very_long']
        quality_data.append(row)
    
    st.dataframe(pd.DataFrame(quality_data), width='stretch', hide_index=True)


def render_sample_viewer(samples: List[Dict], fields: List[str]) -> None:
    """渲染样本查看器"""
    st.markdown("### 🔍 样本查看器")
    
    # 样本选择
    sample_idx = st.slider("选择样本", 0, len(samples) - 1, 0)
    sample = samples[sample_idx]
    
    # 显示样本
    for field in fields:
        if field in sample:
            value = sample[field]
            
            # 根据类型显示
            if isinstance(value, str):
                if len(value) > 500:
                    with st.expander(f"**{field}** ({len(value)} 字符)", expanded=False):
                        st.text(value)
                else:
                    st.markdown(f"**{field}**")
                    st.info(value)
            elif isinstance(value, (list, dict)):
                with st.expander(f"**{field}** ({type(value).__name__})"):
                    st.json(value)
            else:
                st.markdown(f"**{field}**: {value}")
        
        st.markdown("---")


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Dataset 透视镜</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 无需下载全量数据，流式预览 HuggingFace Dataset，快速分析数据分布和质量。
    支持数据统计、分布可视化、质量检查等功能。
    </div>
    """, unsafe_allow_html=True)
    
    # 数据集配置
    st.markdown("### ⚙️ 数据集配置")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        dataset_id = st.text_input(
            "Dataset ID",
            value="tatsu-lab/alpaca",
            placeholder="输入 HuggingFace Dataset ID",
            help="例如: tatsu-lab/alpaca, databricks/dolly-15k"
        )
    
    with col2:
        config_name = st.text_input(
            "Config (可选)",
            value="",
            placeholder="default",
            help="某些数据集有多个配置，如语言子集"
        )
    
    with col3:
        split = st.selectbox(
            "Split",
            ["train", "validation", "test"],
            help="数据集分片"
        )
    
    col_samples, col_action = st.columns([2, 1])
    
    with col_samples:
        num_samples = st.slider(
            "预览样本数",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="流式加载的样本数量"
        )
    
    with col_action:
        st.markdown("<br>", unsafe_allow_html=True)
        load_button = st.button("🚀 加载数据集", type="primary", width='stretch')
    
    if load_button:
        try:
            from datasets import load_dataset
            
            with st.spinner(f"流式加载 {dataset_id}..."):
                # 构建加载参数
                load_kwargs = {
                    "path": dataset_id,
                    "split": split,
                    "streaming": True,
                    "trust_remote_code": True
                }
                
                if config_name:
                    load_kwargs["name"] = config_name
                
                # 尝试加载
                try:
                    ds = load_dataset(**load_kwargs)
                except Exception as e:
                    # 如果 streaming 失败，尝试非 streaming
                    st.warning(f"流式加载失败，尝试标准加载...")
                    load_kwargs["streaming"] = False
                    ds = load_dataset(**load_kwargs)
                    ds = iter(ds)
                
                # 获取样本
                samples = []
                progress_bar = st.progress(0)
                for i, item in enumerate(ds):
                    if i >= num_samples:
                        break
                    samples.append(item)
                    if i % 10 == 0:
                        progress_bar.progress(i / num_samples)
                
                progress_bar.progress(1.0)
            
            if not samples:
                st.error("未能加载任何样本")
                return
            
            st.success(f"✅ 成功加载 {len(samples)} 条样本")
            
            # 保存到 session_state
            st.session_state['dataset_samples'] = samples
            st.session_state['dataset_id'] = dataset_id
            
        except ImportError:
            st.error("请确保已安装 `datasets` 库：`pip install datasets`")
            return
        except Exception as e:
            st.error(f"加载失败: {str(e)}")
            st.info("""
            **常见问题排查**:
            - 检查网络连接
            - 确认 Dataset ID 正确
            - 某些数据集需要指定 config
            - 部分数据集需要登录 HuggingFace
            """)
            return
    
    # 如果有已加载的数据
    if 'dataset_samples' not in st.session_state:
        # 显示常用数据集推荐
        st.markdown("---")
        st.markdown("### 📚 常用数据集推荐")
        
        datasets_info = [
            {"id": "tatsu-lab/alpaca", "desc": "52K 指令微调数据", "type": "SFT"},
            {"id": "databricks/dolly-15k", "desc": "15K 高质量指令数据", "type": "SFT"},
            {"id": "HuggingFaceH4/ultrachat_200k", "desc": "200K 多轮对话", "type": "Chat"},
            {"id": "Open-Orca/OpenOrca", "desc": "大规模推理数据集", "type": "Reasoning"},
            {"id": "garage-bAInd/Open-Platypus", "desc": "高质量 STEM 数据", "type": "SFT"},
            {"id": "teknium/openhermes", "desc": "高质量综合数据", "type": "SFT"},
            {"id": "HuggingFaceFW/fineweb", "desc": "高质量 Web 预训练", "type": "Pretrain"},
            {"id": "allenai/c4", "desc": "Common Crawl 清洗版", "type": "Pretrain"}
        ]
        
        # 分类展示
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 SFT 数据集**")
            for ds in datasets_info:
                if ds['type'] == 'SFT':
                    st.markdown(f"- `{ds['id']}`: {ds['desc']}")
        
        with col2:
            st.markdown("**💬 对话 & 预训练**")
            for ds in datasets_info:
                if ds['type'] in ['Chat', 'Reasoning', 'Pretrain']:
                    st.markdown(f"- `{ds['id']}`: {ds['desc']}")
        
        return
    
    # 有数据时显示分析
    samples = st.session_state['dataset_samples']
    df = pd.DataFrame(samples)
    fields = list(samples[0].keys())
    
    # 创建 tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据概览", "📈 分布分析", "📋 质量检查", "🔍 样本浏览"])
    
    with tab1:
        st.markdown("### 字段结构")
        
        # 字段信息表格
        field_info = []
        for field in fields:
            sample_val = samples[0][field]
            val_type = type(sample_val).__name__
            
            # 示例值
            if isinstance(sample_val, str):
                preview = sample_val[:100] + '...' if len(sample_val) > 100 else sample_val
            elif isinstance(sample_val, (list, dict)):
                preview = str(sample_val)[:100] + '...'
            else:
                preview = str(sample_val)
            
            field_info.append({
                '字段名': field,
                '类型': val_type,
                '示例值': preview
            })
        
        st.dataframe(
            pd.DataFrame(field_info), 
            width='stretch',
            hide_index=True,
            column_config={
                "示例值": st.column_config.TextColumn(width="large")
            }
        )
        
        # 字段统计
        st.markdown("### 字段统计")
        
        field_stats = []
        for field in fields:
            values = [s[field] for s in samples]
            stats = analyze_field_content(values, field)
            field_stats.append(stats)
        
        # 显示统计表格
        stats_df = pd.DataFrame([{
            '字段': s['name'],
            '类型': s['type'],
            '空值数': s['null_count'],
            '平均长度/值': f"{s['avg_length']:.0f}" if s['avg_length'] > 0 else '-',
            '最小': f"{s['min_length']:.0f}" if s['min_length'] > 0 else '-',
            '最大': f"{s['max_length']:.0f}" if s['max_length'] > 0 else '-'
        } for s in field_stats])
        
        st.dataframe(stats_df, width='stretch', hide_index=True)
        
        # 汇总图表
        fig = render_field_stats_chart(field_stats)
        if fig:
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.markdown("### 分布分析")
        
        # 选择要分析的字段
        string_fields = [f for f in fields if isinstance(samples[0].get(f), str)]
        
        if not string_fields:
            st.info("没有找到文本字段用于分析")
        else:
            selected_field = st.selectbox(
                "选择分析字段",
                string_fields,
                help="选择一个文本字段进行分布分析"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 长度分布
                fig_len = render_length_distribution(df, selected_field)
                st.plotly_chart(fig_len, width='stretch')
            
            with col2:
                # 词数分布
                fig_word = render_word_count_distribution(df, selected_field)
                st.plotly_chart(fig_word, width='stretch')
            
            # 详细统计
            st.markdown("#### 详细统计")
            
            lengths = df[selected_field].astype(str).str.len()
            word_counts = df[selected_field].astype(str).str.split().str.len()
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("平均字符数", f"{lengths.mean():.0f}")
            with col_b:
                st.metric("中位数字符数", f"{lengths.median():.0f}")
            with col_c:
                st.metric("平均词数", f"{word_counts.mean():.0f}")
            with col_d:
                st.metric("最长样本", f"{lengths.max():.0f} 字符")
            
            # Token 估算
            st.markdown("#### Token 数估算")
            st.caption("粗略估算，实际数量取决于具体 tokenizer")
            
            # 假设平均 4 字符 = 1 token (英文)
            # 中文大约 1.5-2 字符 = 1 token
            avg_tokens_en = lengths.mean() / 4
            avg_tokens_zh = lengths.mean() / 1.5
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("估算 Token (英文模型)", f"~{avg_tokens_en:.0f}")
            with col2:
                st.metric("估算 Token (中文模型)", f"~{avg_tokens_zh:.0f}")
    
    with tab3:
        # 数据质量检查
        quality = check_data_quality(df)
        render_quality_report(quality)
        
        # 问题样本展示
        st.markdown("### ⚠️ 潜在问题样本")
        
        problem_samples = []
        
        for field in fields:
            if df[field].dtype == 'object':
                # 找出空值或很短的样本
                short_mask = df[field].astype(str).str.len() < 10
                empty_mask = df[field].astype(str).str.strip() == ''
                
                for idx in df[short_mask | empty_mask].index[:3]:
                    problem_samples.append({
                        'index': idx,
                        'field': field,
                        'issue': '内容过短或为空',
                        'value': str(df.loc[idx, field])[:100]
                    })
        
        if problem_samples:
            st.dataframe(
                pd.DataFrame(problem_samples),
                width='stretch',
                hide_index=True
            )
        else:
            st.success("✅ 未发现明显的数据质量问题")
    
    with tab4:
        render_sample_viewer(samples, fields)
        
        # 原始 JSON 查看
        st.markdown("### 📄 原始 JSON")
        
        json_idx = st.number_input(
            "样本索引",
            min_value=0,
            max_value=len(samples) - 1,
            value=0
        )
        
        st.json(samples[json_idx])
        
        # 导出功能
        st.markdown("### 💾 数据导出")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON 导出
            json_str = json.dumps(samples, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 下载 JSON",
                data=json_str,
                file_name=f"{st.session_state.get('dataset_id', 'dataset').replace('/', '_')}_samples.json",
                mime="application/json"
            )
        
        with col2:
            # CSV 导出 (仅适用于扁平结构)
            try:
                csv_str = df.to_csv(index=False)
                st.download_button(
                    "📥 下载 CSV",
                    data=csv_str,
                    file_name=f"{st.session_state.get('dataset_id', 'dataset').replace('/', '_')}_samples.csv",
                    mime="text/csv"
                )
            except:
                st.caption("CSV 导出不适用于复杂嵌套结构")
