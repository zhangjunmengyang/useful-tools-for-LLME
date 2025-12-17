"""
Dataset Viewer - HuggingFace Dataset 流式预览与分析
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import json
import html


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
    
    non_null = [v for v in values if v is not None and v != '']
    stats['null_count'] = len(values) - len(non_null)
    
    if not non_null:
        return stats
    
    first_val = non_null[0]
    if isinstance(first_val, str):
        stats['type'] = 'string'
        lengths = [len(str(v)) for v in non_null]
        stats['avg_length'] = np.mean(lengths)
        stats['min_length'] = np.min(lengths)
        stats['max_length'] = np.max(lengths)
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
    mean_val = lengths.mean()
    median_val = lengths.median()
    
    fig = go.Figure(data=go.Histogram(
        x=lengths,
        nbinsx=30,
        marker_color='#2563EB',
        opacity=0.8,
        name='Distribution'
    ))

    fig.add_vline(x=mean_val, line_dash="dash", line_color="#DC2626", line_width=2)
    fig.add_vline(x=median_val, line_dash="dot", line_color="#059669", line_width=2)

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(dash='dash', color='#DC2626', width=2),
        name=f'Mean: {mean_val:.0f}'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(dash='dot', color='#059669', width=2),
        name=f'Median: {median_val:.0f}'
    ))

    fig.update_layout(
        title=f"'{field}' Length Distribution",
        xaxis_title="Character Length",
        yaxis_title="Sample Count",
        height=400,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True
    )
    
    return fig


def render_word_count_distribution(df: pd.DataFrame, field: str) -> go.Figure:
    """渲染字段词数分布"""
    word_counts = df[field].astype(str).str.split().str.len()
    mean_val = word_counts.mean()
    median_val = word_counts.median()
    
    fig = go.Figure(data=go.Histogram(
        x=word_counts,
        nbinsx=30,
        marker_color='#7C3AED',
        opacity=0.8,
        name='Distribution'
    ))

    fig.add_vline(x=mean_val, line_dash="dash", line_color="#DC2626", line_width=2)
    fig.add_vline(x=median_val, line_dash="dot", line_color="#059669", line_width=2)

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(dash='dash', color='#DC2626', width=2),
        name=f'Mean: {mean_val:.0f}'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(dash='dot', color='#059669', width=2),
        name=f'Median: {median_val:.0f}'
    ))

    fig.update_layout(
        title=f"'{field}' Word Count Distribution (space-separated)",
        xaxis_title="Word Count",
        yaxis_title="Sample Count",
        height=400,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True
    )
    
    return fig


def render_field_stats_chart(field_stats: List[Dict]) -> Optional[go.Figure]:
    """渲染字段统计汇总图"""
    string_fields = [f for f in field_stats if f['type'] == 'string']
    
    if not string_fields:
        return None
    
    names = [f['name'] for f in string_fields]
    avg_lengths = [f['avg_length'] for f in string_fields]
    null_rates = [f['null_count'] / f['total'] * 100 for f in string_fields]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average Length',
        x=names,
        y=avg_lengths,
        marker_color='#2563EB',
        yaxis='y',
        offsetgroup=0
    ))

    fig.add_trace(go.Bar(
        name='Null Rate (%)',
        x=names,
        y=null_rates,
        marker_color='#DC2626',
        yaxis='y2',
        offsetgroup=1
    ))

    fig.update_layout(
        title="Field Statistics Summary",
        xaxis_title="Field",
        yaxis=dict(title="Average Length (chars)", side='left'),
        yaxis2=dict(title="Null Rate (%)", side='right', overlaying='y', range=[0, 100]),
        barmode='group',
        height=450,
        autosize=True,
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
        
        if df[col].dtype == 'object':
            col_quality['very_short'] = (df[col].astype(str).str.len() < 10).sum()
            col_quality['very_long'] = (df[col].astype(str).str.len() > 10000).sum()
        
        quality['fields'][col] = col_quality
    
    return quality


# 全局状态存储
_dataset_cache = {
    'samples': None,
    'field_stats': None,
    'dataset_id': None
}


def load_dataset_samples(dataset_id: str, config_name: str, split: str, num_samples: int, progress=gr.Progress()):
    """加载数据集样本"""
    try:
        from datasets import load_dataset
        
        load_kwargs = {
            "path": dataset_id,
            "split": split,
            "streaming": True,
            "trust_remote_code": True
        }
        
        if config_name and config_name.strip():
            load_kwargs["name"] = config_name.strip()
        
        progress(0.1, desc="Connecting to dataset...")
        
        try:
            ds = load_dataset(**load_kwargs)
        except Exception:
            load_kwargs["streaming"] = False
            ds = load_dataset(**load_kwargs)
            ds = iter(ds)
        
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(item)
            if i % 10 == 0:
                progress((i + 1) / num_samples, desc=f"Loaded {i + 1}/{num_samples} samples...")

        if not samples:
            return None, "Failed to load any samples"
        
        # 计算字段统计
        fields = list(samples[0].keys())
        field_stats = []
        for field in fields:
            values = [s[field] for s in samples]
            stats = analyze_field_content(values, field)
            field_stats.append(stats)
        
        # 缓存数据
        _dataset_cache['samples'] = samples
        _dataset_cache['field_stats'] = field_stats
        _dataset_cache['dataset_id'] = dataset_id
        
        return samples, f"Successfully loaded {len(samples)} samples"

    except ImportError:
        return None, "Please ensure datasets library is installed: pip install datasets"
    except Exception as e:
        return None, f"Loading failed: {str(e)}"


def get_sample_display(sample_idx: int):
    """获取单个样本的显示内容"""
    samples = _dataset_cache.get('samples')
    if not samples:
        return "<div style='height: 500px; display: flex; align-items: center; justify-content: center; color: #6b7280;'>Please load a dataset first</div>", "{}"
    
    sample_idx = max(0, min(sample_idx, len(samples) - 1))
    sample = samples[sample_idx]
    
    # 构建 HTML 显示 - 使用固定高度容器
    html_parts = [
        '<div style="height: 500px; overflow-y: auto; padding-right: 8px;">'
    ]
    for field, value in sample.items():
        escaped_field = html.escape(str(field))
        
        if isinstance(value, str):
            char_count = len(value)
            escaped_value = html.escape(value[:2000]) + ('...' if len(value) > 2000 else '')
            html_parts.append(f"""
            <div style="background: #fff; border: 1px solid #d0d7de; border-radius: 6px; margin-bottom: 12px; overflow: hidden;">
                <div style="padding: 8px 12px; background: #f6f8fa; border-bottom: 1px solid #d0d7de; font-size: 12px; font-weight: 600; font-family: monospace;">
                    {escaped_field} <span style="color: #656d76; font-weight: 400; margin-left: 8px;">{char_count} chars</span>
                </div>
                <div style="padding: 12px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word;">{escaped_value}</div>
            </div>
            """)
        elif isinstance(value, (list, dict)):
            html_parts.append(f"""
            <div style="background: #fff; border: 1px solid #d0d7de; border-radius: 6px; margin-bottom: 12px; overflow: hidden;">
                <div style="padding: 8px 12px; background: #f6f8fa; border-bottom: 1px solid #d0d7de; font-size: 12px; font-weight: 600; font-family: monospace;">
                    {escaped_field} <span style="color: #656d76; font-weight: 400; margin-left: 8px;">{type(value).__name__}</span>
                </div>
                <div style="padding: 12px; font-size: 13px; font-family: monospace; background: #f8f9fa;">
                    <pre style="margin: 0; white-space: pre-wrap;">{html.escape(json.dumps(value, ensure_ascii=False, indent=2)[:1000])}</pre>
                </div>
            </div>
            """)
        else:
            escaped_value = html.escape(str(value))
            html_parts.append(f"""
            <div style="background: #fff; border: 1px solid #d0d7de; border-radius: 6px; margin-bottom: 12px; overflow: hidden;">
                <div style="padding: 8px 12px; background: #f6f8fa; border-bottom: 1px solid #d0d7de; font-size: 12px; font-weight: 600; font-family: monospace;">
                    {escaped_field}
                </div>
                <div style="padding: 12px; font-size: 14px;">{escaped_value}</div>
            </div>
            """)
    
    html_parts.append('</div>')
    sample_html = "".join(html_parts)
    json_str = json.dumps(sample, ensure_ascii=False, indent=2)
    
    return sample_html, json_str


def get_statistics_data():
    """获取统计数据"""
    samples = _dataset_cache.get('samples')
    field_stats = _dataset_cache.get('field_stats')
    
    if not samples or not field_stats:
        return None, None, None
    
    df = pd.DataFrame(samples)
    fields = list(samples[0].keys())
    
    # 字段信息表
    field_info = []
    for field in fields:
        sample_val = samples[0][field]
        val_type = type(sample_val).__name__
        
        if isinstance(sample_val, str):
            preview = sample_val[:100] + '...' if len(sample_val) > 100 else sample_val
        elif isinstance(sample_val, (list, dict)):
            preview = str(sample_val)[:100] + '...'
        else:
            preview = str(sample_val)
        
        field_info.append({
            'Field Name': field,
            'Type': val_type,
            'Sample Value': preview
        })
    
    # 统计表
    stats_data = []
    for s in field_stats:
        stats_data.append({
            'Field': s['name'],
            'Type': s['type'],
            'Null Count': s['null_count'],
            'Avg Length/Value': f"{s['avg_length']:.0f}" if s['avg_length'] > 0 else '-',
            'Min': f"{s['min_length']:.0f}" if s['min_length'] > 0 else '-',
            'Max': f"{s['max_length']:.0f}" if s['max_length'] > 0 else '-'
        })
    
    return pd.DataFrame(field_info), pd.DataFrame(stats_data), render_field_stats_chart(field_stats)


def get_distribution_data(field: str):
    """获取分布分析数据"""
    samples = _dataset_cache.get('samples')
    if not samples:
        return None, None, ""
    
    df = pd.DataFrame(samples)
    
    if field not in df.columns:
        return None, None, ""
    
    lengths = df[field].astype(str).str.len()
    word_counts = df[field].astype(str).str.split().str.len()
    
    stats_text = f"""
    **Detailed Statistics**
    - Average character count: {lengths.mean():.0f}
    - Median character count: {lengths.median():.0f}
    - Average word count: {word_counts.mean():.0f}
    - Longest sample: {lengths.max():.0f} characters

    **Token Count Estimate** (rough estimate)
    - English model: ~{lengths.mean() / 4:.0f} tokens
    - Chinese model: ~{lengths.mean() / 1.5:.0f} tokens
    """
    
    return render_length_distribution(df, field), render_word_count_distribution(df, field), stats_text


def get_quality_data():
    """获取质量检查数据"""
    samples = _dataset_cache.get('samples')
    if not samples:
        return "", None, None
    
    df = pd.DataFrame(samples)
    quality = check_data_quality(df)
    
    # 计算健康度
    total_issues = quality['duplicate_count']
    for field_quality in quality['fields'].values():
        total_issues += field_quality['null_count']
    health_score = max(0, 100 - (total_issues / quality['total_samples'] * 10))
    
    dup_rate = quality['duplicate_count'] / quality['total_samples'] * 100
    
    summary = f"""
    **Data Overview**
    - Total samples: {quality['total_samples']}
    - Duplicate samples: {quality['duplicate_count']} ({dup_rate:.1f}%)
    - Health score: {health_score:.0f}/100
    """
    
    # 字段质量表
    quality_data = []
    for field, field_quality in quality['fields'].items():
        row = {
            'Field': field,
            'Null Count': field_quality['null_count'],
            'Null Rate': f"{field_quality['null_rate']:.1f}%",
            'Empty Strings': field_quality['empty_string_count']
        }
        if 'very_short' in field_quality:
            row['Too Short (<10 chars)'] = field_quality['very_short']
        if 'very_long' in field_quality:
            row['Too Long (>10k chars)'] = field_quality['very_long']
        quality_data.append(row)
    
    # 问题样本
    problem_samples = []
    fields = list(samples[0].keys())
    for field in fields:
        if df[field].dtype == 'object':
            short_mask = df[field].astype(str).str.len() < 10
            empty_mask = df[field].astype(str).str.strip() == ''
            
            for idx in df[short_mask | empty_mask].index[:3]:
                problem_samples.append({
                    'index': idx,
                    'field': field,
                    'issue': 'Content too short or empty',
                    'value': str(df.loc[idx, field])[:100]
                })
    
    problem_df = pd.DataFrame(problem_samples) if problem_samples else None
    
    return summary, pd.DataFrame(quality_data), problem_df


def get_string_fields():
    """获取文本字段列表"""
    samples = _dataset_cache.get('samples')
    if not samples:
        return []
    return [f for f in samples[0].keys() if isinstance(samples[0].get(f), str)]


def render():
    """渲染页面"""

    # 数据集配置区
    with gr.Group():
        with gr.Row():
            dataset_id = gr.Textbox(
                label="Dataset ID",
                value="tatsu-lab/alpaca",
                placeholder="e.g., tatsu-lab/alpaca, databricks/dolly-15k"
            )
            config_name = gr.Textbox(
                label="Config (Optional)",
                value="",
                placeholder="default"
            )
            split = gr.Dropdown(
                label="Split",
                choices=["train", "validation", "test"],
                value="train"
            )
        
        with gr.Row():
            num_samples = gr.Slider(
                label="Sample Count",
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                scale=3
            )
            load_btn = gr.Button("Load Dataset", variant="primary", scale=1, min_width=150)
        
        load_status = gr.Markdown("")
    
    # Tab 页面
    with gr.Tabs():
        # Tab 1: 样本预览
        with gr.Tab("Sample Preview"):
            with gr.Row():
                sample_idx = gr.Slider(
                    label="Sample Index",
                    minimum=0,
                    maximum=99,
                    value=0,
                    step=1,
                    interactive=True
                )

            # 固定高度容器防止滑动时跳动
            sample_display = gr.HTML(
                label="Sample Content",
                elem_id="sample-display-container"
            )

            with gr.Accordion("Raw JSON", open=False):
                json_display = gr.Code(language="json", label="JSON")
        
        # Tab 2: 数据统计
        with gr.Tab("Statistics"):
            gr.Markdown("### Field Structure")
            field_info_table = gr.Dataframe(label="Field Info")

            gr.Markdown("### Field Statistics")
            stats_table = gr.Dataframe(label="Stats Data")
            stats_chart = gr.Plot(label="Stats Chart")

            gr.Markdown("### Distribution Analysis")
            field_select = gr.Dropdown(label="Select Field", choices=[], interactive=True)

            with gr.Row():
                length_chart = gr.Plot(label="Length Distribution")
                word_chart = gr.Plot(label="Word Count Distribution")

            dist_stats = gr.Markdown("")
        
        # Tab 3: 质量检查
        with gr.Tab("Quality Check"):
            quality_summary = gr.Markdown("")

            gr.Markdown("### Field Quality Details")
            quality_table = gr.Dataframe(label="Quality Data")

            gr.Markdown("### Potential Problem Samples")
            problem_table = gr.Dataframe(label="Problem Samples")
    
    # 事件绑定
    def on_load_dataset(dataset_id, config_name, split, num_samples):
        samples, message = load_dataset_samples(dataset_id, config_name, split, int(num_samples))
        if samples:
            html, json_str = get_sample_display(0)
            max_idx = len(samples) - 1
            
            # 自动计算统计数据
            field_info, stats_data, chart = get_statistics_data()
            string_fields = get_string_fields()
            first_field = string_fields[0] if string_fields else None
            
            # 自动计算第一个字段的分布
            length_fig, word_fig, dist_text = (None, None, "")
            if first_field:
                length_fig, word_fig, dist_text = get_distribution_data(first_field)
            
            # 自动计算质量数据
            quality_sum, quality_df, problem_df = get_quality_data()
            
            return (
                message,
                gr.update(maximum=max_idx, value=0),
                html,
                json_str,
                field_info,
                stats_data,
                chart,
                gr.update(choices=string_fields, value=first_field),
                length_fig,
                word_fig,
                dist_text,
                quality_sum,
                quality_df,
                problem_df
            )
        return (message, gr.update(), "", "{}", 
                None, None, None, gr.update(choices=[], value=None),
                None, None, "", "", None, None)
    
    def on_sample_change(idx):
        html, json_str = get_sample_display(int(idx))
        return html, json_str
    
    def on_field_change(field):
        if not field:
            return None, None, ""
        return get_distribution_data(field)
    
    load_btn.click(
        fn=on_load_dataset,
        inputs=[dataset_id, config_name, split, num_samples],
        outputs=[load_status, sample_idx, sample_display, json_display,
                 field_info_table, stats_table, stats_chart, field_select,
                 length_chart, word_chart, dist_stats,
                 quality_summary, quality_table, problem_table]
    )
    
    sample_idx.change(
        fn=on_sample_change,
        inputs=[sample_idx],
        outputs=[sample_display, json_display]
    )
    
    field_select.change(
        fn=on_field_change,
        inputs=[field_select],
        outputs=[length_chart, word_chart, dist_stats]
    )
