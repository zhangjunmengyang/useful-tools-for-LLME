"""
数据清洗 - 测试清洗规则和 PPL 过滤
"""

import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_lab.data_utils import (
    CLEANING_RULES, 
    clean_text, 
    normalize_unicode,
    PPL_MODELS,
    calculate_perplexity,
    batch_calculate_ppl,
    filter_by_ppl,
    get_ppl_quality_label
)


def render_ppl_histogram(ppl_values: list, threshold_max: float) -> go.Figure:
    """渲染 PPL 分布直方图"""
    # 过滤掉无穷大值
    valid_values = [v for v in ppl_values if v != float('inf') and v < 10000]
    
    if not valid_values:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=valid_values,
        nbinsx=30,
        marker_color='#2563EB',
        opacity=0.7,
        name='PPL 分布'
    ))
    
    # 添加阈值线
    fig.add_vline(
        x=threshold_max, 
        line_dash="dash", 
        line_color="#DC2626",
        annotation_text=f"阈值: {threshold_max}"
    )
    
    # 添加参考线
    reference_lines = [
        (50, "优秀", "#059669"),
        (100, "良好", "#2563EB"),
        (300, "一般", "#D97706"),
    ]
    
    for val, label, color in reference_lines:
        if val < max(valid_values):
            fig.add_vline(
                x=val,
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top"
            )
    
    fig.update_layout(
        title="PPL 分布直方图",
        xaxis_title="Perplexity",
        yaxis_title="样本数",
        height=350,
        showlegend=False
    )
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">数据清洗</h1>', unsafe_allow_html=True)
    
    
    # 创建 tabs
    tab1, tab2 = st.tabs(["规则清洗", "PPL 过滤"])
    
    with tab1:
        # 清洗规则选择 - 置顶
        st.markdown("### 清洗规则")
        
        selected_rules = []
        rule_cols = st.columns(3)
        for idx, (rule_id, rule_info) in enumerate(CLEANING_RULES.items()):
            with rule_cols[idx % 3]:
                if st.checkbox(rule_info['name'], value=True, key=f"rule_{rule_id}"):
                    selected_rules.append(rule_id)
        
        # Unicode 规范化和自定义正则
        col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
        with col_opt1:
            unicode_form = st.selectbox("Unicode 规范化", ["无", "NFC", "NFD", "NFKC", "NFKD"])
        with col_opt2:
            custom_pattern = st.text_input("自定义正则 Pattern", placeholder=r"如 \d+")
        with col_opt3:
            custom_replacement = st.text_input("替换为", placeholder="替换文本")
        
        st.markdown("---")
        
        # 输入输出两列
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 输入 (脏数据)")
            dirty_text = st.text_area(
                "原始文本",
                value="""<p>这是一段 HTML 文本</p>
访问 https://example.com 了解更多
联系邮箱: test@email.com
包含   多余   空格
特殊符号★☆♠♣""",
                height=200,
                key="dirty_input"
            )
        
        # 应用清洗（在列外计算，确保规则生效）
        cleaned = dirty_text
        
        # 应用选中的规则
        for rule_id in selected_rules:
            rule = CLEANING_RULES[rule_id]
            cleaned = re.sub(rule['pattern'], rule['replacement'], cleaned)
        
        # Unicode 规范化
        if unicode_form != "无":
            cleaned = normalize_unicode(cleaned, unicode_form)
        
        # 自定义正则
        if custom_pattern:
            try:
                cleaned = re.sub(custom_pattern, custom_replacement, cleaned)
            except re.error as e:
                st.error(f"正则错误: {e}")
        
        cleaned = cleaned.strip()
        
        with col_right:
            st.markdown("### 输出 (清洗后)")
            st.text_area("清洗结果", value=cleaned, height=200, disabled=True)
            
            # 统计
            st.markdown("### 统计")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("原始长度", len(dirty_text))
            with stat_col2:
                st.metric("清洗后长度", len(cleaned))
            with stat_col3:
                reduction = (1 - len(cleaned) / len(dirty_text)) * 100 if dirty_text else 0
                st.metric("缩减比例", f"{reduction:.1f}%")
    
    with tab2:
        # 模型选择
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model_choice = st.selectbox(
                "选择 PPL 计算模型",
                options=list(PPL_MODELS.keys()),
                help="选择用于计算 PPL 的语言模型"
            )
        
        with col2:
            model_info = PPL_MODELS[model_choice]
        # 阈值设置
        col_a, col_b = st.columns(2)
        with col_a:
            min_ppl = st.number_input("最小 PPL", value=0.0, min_value=0.0, help="PPL 低于此值的文本会被过滤（可能是重复/无意义）")
        with col_b:
            max_ppl = st.number_input("最大 PPL", value=500.0, min_value=1.0, help="PPL 高于此值的文本会被过滤")
        
        st.markdown("---")
        
        # 输入模式选择
        input_mode = st.radio("输入模式", ["单条文本", "批量文本"], horizontal=True)
        
        if input_mode == "单条文本":
            # 单条文本模式
            text_input = st.text_area(
                "输入文本",
                value="The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity calculation.",
                height=150,
                placeholder="输入要计算 PPL 的文本..."
            )
            
            if st.button("计算 PPL", type="primary"):
                if not text_input.strip():
                    st.warning("请输入文本")
                else:
                    try:
                        import torch
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        
                        with st.spinner(f"加载模型 {model_choice}..."):
                            tokenizer = AutoTokenizer.from_pretrained(model_info['id'])
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['id'],
                                torch_dtype=torch.float32,
                                device_map="cpu"
                            )
                            model.eval()
                        
                        with st.spinner("计算 PPL..."):
                            ppl, details = calculate_perplexity(text_input, model, tokenizer)
                        
                        # 显示结果
                        label, color = get_ppl_quality_label(ppl)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Perplexity", f"{ppl:.2f}")
                        with col2:
                            st.metric("质量评级", label)
                        with col3:
                            st.metric("序列长度", details.get('seq_length', 'N/A'))
                        
                        # 可视化评级
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, {color}22, transparent); 
                                    padding: 15px; border-radius: 8px; border-left: 4px solid {color};">
                            <span style="color: {color}; font-size: 24px; font-weight: bold;">
                                PPL = {ppl:.2f}
                            </span>
                            <span style="margin-left: 20px; color: #6B7280;">
                                质量: <b>{label}</b>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 阈值判断
                        if min_ppl <= ppl <= max_ppl:
                            st.info(f"通过过滤 (PPL 在 {min_ppl} - {max_ppl} 范围内)")
                        else:
                            st.warning(f"被过滤 (PPL 超出 {min_ppl} - {max_ppl} 范围)")
                        
                        # 详细信息
                        with st.expander("详细信息"):
                            st.json(details)
                    
                    except ImportError:
                        st.error("请确保已安装 `transformers` 和 `torch` 库")
                    except Exception as e:
                        st.error(f"计算失败: {str(e)}")
        
        else:
            # 批量文本模式
            st.markdown("每行一条文本:")
            
            batch_input = st.text_area(
                "批量输入",
                value="""The weather is nice today.
This is a normal English sentence.
asdfjkl qwerty random gibberish text
机器学习是人工智能的一个重要分支。
!!!@@@###$$$%%%^^^&&&***
She went to the store to buy some groceries for dinner.""",
                height=200,
                placeholder="每行一条文本..."
            )
            
            if st.button("批量计算 PPL", type="primary"):
                texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
                
                if not texts:
                    st.warning("请输入文本")
                else:
                    try:
                        import torch
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        
                        with st.spinner(f"加载模型 {model_choice}..."):
                            tokenizer = AutoTokenizer.from_pretrained(model_info['id'])
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['id'],
                                torch_dtype=torch.float32,
                                device_map="cpu"
                            )
                            model.eval()
                        
                        # 批量计算
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, text in enumerate(texts):
                            ppl, details = calculate_perplexity(text, model, tokenizer)
                            label, color = get_ppl_quality_label(ppl)
                            accepted = min_ppl <= ppl <= max_ppl
                            
                            results.append({
                                "文本": text[:50] + "..." if len(text) > 50 else text,
                                "PPL": ppl,
                                "评级": label,
                                "通过": "Yes" if accepted else "No",
                                "长度": len(text)
                            })
                            
                            progress_bar.progress((i + 1) / len(texts))
                        
                        progress_bar.empty()
                        
                        # 统计
                        ppl_values = [r["PPL"] for r in results]
                        accepted_count = sum(1 for r in results if r["通过"] == "Yes")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总样本数", len(results))
                        with col2:
                            st.metric("通过数", accepted_count)
                        with col3:
                            st.metric("过滤数", len(results) - accepted_count)
                        with col4:
                            valid_ppl = [p for p in ppl_values if p != float('inf')]
                            avg_ppl = np.mean(valid_ppl) if valid_ppl else 0
                            st.metric("平均 PPL", f"{avg_ppl:.1f}")
                        
                        # 分布图
                        fig = render_ppl_histogram(ppl_values, max_ppl)
                        if fig:
                            st.plotly_chart(fig, width='stretch')
                        
                        # 结果表格
                        st.markdown("### 详细结果")
                        df = pd.DataFrame(results)
                        st.dataframe(df, width='stretch', hide_index=True)
                        
                        # 显示被过滤的文本
                        rejected = [r for r in results if r["通过"] == "No"]
                        if rejected:
                            with st.expander(f"被过滤的文本 ({len(rejected)} 条)"):
                                for r in rejected:
                                    st.markdown(f"- **PPL={r['PPL']:.1f}** ({r['评级']}): {r['文本']}")
                    
                    except ImportError:
                        st.error("请确保已安装 `transformers` 和 `torch` 库")
                    except Exception as e:
                        st.error(f"计算失败: {str(e)}")
