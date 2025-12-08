"""
Beam Search 可视化 - 展示搜索路径树
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any
import torch
from generation_lab.generation_utils import (
    DEMO_MODELS,
    load_model_and_tokenizer
)


def beam_search_with_history(
    model: Any,
    tokenizer: Any,
    prompt: str,
    beam_size: int = 3,
    max_steps: int = 5,
    eos_token_id: int = None
) -> Dict:
    """
    执行 Beam Search 并记录历史
    
    Returns:
        包含每步搜索历史的字典
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    # 初始化
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_length = input_ids.shape[1]
    
    # 初始 beam
    beams = [{
        'ids': input_ids[0].tolist(),
        'score': 0.0,
        'text': prompt,
        'tokens': [],
        'finished': False
    }]
    
    history = {
        'prompt': prompt,
        'beam_size': beam_size,
        'steps': []
    }
    
    for step in range(max_steps):
        step_record = {
            'step': step,
            'active_beams': [],
            'all_candidates': [],
            'pruned': []
        }
        
        all_candidates = []
        
        for beam_idx, beam in enumerate(beams):
            if beam['finished']:
                all_candidates.append({
                    **beam,
                    'new_token': '[EOS]',
                    'new_token_id': eos_token_id,
                    'token_score': 0.0,
                    'parent_idx': beam_idx,
                    'action': 'finished'
                })
                continue
            
            # 获取下一个 token 的分布
            current_ids = torch.tensor([beam['ids']])
            
            with torch.no_grad():
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 获取 top-k 候选
            top_log_probs, top_indices = torch.topk(log_probs, beam_size * 2)
            
            for i in range(beam_size * 2):
                token_id = top_indices[i].item()
                token_log_prob = top_log_probs[i].item()
                token_str = tokenizer.decode([token_id])
                
                new_ids = beam['ids'] + [token_id]
                new_score = beam['score'] + token_log_prob
                new_text = tokenizer.decode(new_ids[prompt_length:])
                
                candidate = {
                    'ids': new_ids,
                    'score': new_score,
                    'text': prompt + new_text,
                    'tokens': beam['tokens'] + [token_str],
                    'finished': token_id == eos_token_id,
                    'new_token': token_str,
                    'new_token_id': token_id,
                    'token_score': token_log_prob,
                    'parent_idx': beam_idx,
                    'parent_text': ' '.join(beam['tokens']) if beam['tokens'] else '[START]'
                }
                all_candidates.append(candidate)
        
        # 排序并选择 top beam_size
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 记录所有候选
        step_record['all_candidates'] = [{
            'text': c.get('new_token', ''),
            'token_id': c.get('new_token_id', -1),
            'cumulative_score': c['score'],
            'token_score': c.get('token_score', 0),
            'parent': c.get('parent_text', ''),
            'full_sequence': ' '.join(c['tokens'])
        } for c in all_candidates]
        
        # 选择保留的 beams
        selected = all_candidates[:beam_size]
        pruned = all_candidates[beam_size:]
        
        step_record['active_beams'] = [{
            'text': s.get('new_token', ''),
            'full_sequence': ' '.join(s['tokens']),
            'cumulative_score': s['score'],
            'probability': np.exp(s['score'])
        } for s in selected]
        
        step_record['pruned'] = [{
            'text': p.get('new_token', ''),
            'full_sequence': ' '.join(p['tokens']),
            'cumulative_score': p['score'],
            'reason': 'score_too_low'
        } for p in pruned[:beam_size]]  # 只记录前几个被剪枝的
        
        history['steps'].append(step_record)
        
        # 更新 beams
        beams = [{
            'ids': s['ids'],
            'score': s['score'],
            'text': s['text'],
            'tokens': s['tokens'],
            'finished': s['finished']
        } for s in selected]
        
        # 检查是否所有 beam 都结束
        if all(b['finished'] for b in beams):
            break
    
    # 最终结果
    history['final_beams'] = [{
        'rank': i + 1,
        'sequence': ' '.join(b['tokens']),
        'score': b['score'],
        'probability': np.exp(b['score']),
        'finished': b['finished']
    } for i, b in enumerate(beams)]
    
    return history


def render_beam_tree(history: Dict) -> go.Figure:
    """渲染 Beam Search 树形图"""
    fig = go.Figure()
    
    beam_size = history['beam_size']
    steps = history['steps']
    
    if not steps:
        return fig
    
    # 构建节点
    nodes_x = []
    nodes_y = []
    nodes_text = []
    nodes_color = []
    
    edges_x = []
    edges_y = []
    
    # 起始节点
    nodes_x.append(0)
    nodes_y.append(0)
    nodes_text.append("[START]")
    nodes_color.append('#2563EB')
    
    # 记录每一步每个 beam 的位置
    node_positions = {(0, 0): 0}  # (step, beam_idx) -> node_idx
    
    for step_idx, step in enumerate(steps):
        active_beams = step['active_beams']
        
        for beam_idx, beam in enumerate(active_beams):
            # 计算节点位置
            x = step_idx + 1
            y = (beam_idx - len(active_beams) / 2 + 0.5) * 2
            
            node_idx = len(nodes_x)
            node_positions[(step_idx + 1, beam_idx)] = node_idx
            
            nodes_x.append(x)
            nodes_y.append(y)
            nodes_text.append(f"{beam['text']}<br>Score: {beam['cumulative_score']:.2f}")
            nodes_color.append('#2563EB' if beam_idx == 0 else '#60A5FA')
            
            # 添加边 (从上一步连接)
            if step_idx == 0:
                parent_idx = 0  # 从起始节点
            else:
                parent_idx = node_positions.get((step_idx, 0), 0)  # 简化处理
            
            edges_x.extend([nodes_x[parent_idx], x, None])
            edges_y.extend([nodes_y[parent_idx], y, None])
    
    # 绘制边
    fig.add_trace(go.Scatter(
        x=edges_x,
        y=edges_y,
        mode='lines',
        line=dict(color='#E5E7EB', width=2),
        hoverinfo='none'
    ))
    
    # 绘制节点
    fig.add_trace(go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        marker=dict(size=30, color=nodes_color),
        text=[t.split('<br>')[0] for t in nodes_text],
        textposition='top center',
        hovertext=nodes_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Beam Search 搜索树",
        showlegend=False,
        xaxis=dict(
            title="Step",
            showgrid=False,
            zeroline=False,
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def render_step_detail(step_data: Dict) -> None:
    """渲染单步详情"""
    st.markdown(f"#### Step {step_data['step'] + 1}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ 保留的 Beam**")
        for i, beam in enumerate(step_data['active_beams']):
            score_color = '#059669' if i == 0 else '#2563EB'
            st.markdown(f"""
            <div style="background: #F3F4F6; padding: 10px; border-radius: 6px; margin: 5px 0;
                        border-left: 3px solid {score_color};">
                <b>Beam {i+1}</b>: "{beam['text']}"<br>
                <small>序列: {beam['full_sequence'][:50]}...</small><br>
                <small>累积分数: {beam['cumulative_score']:.4f} | 
                       概率: {beam['probability']:.2e}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**❌ 被剪枝的候选**")
        if step_data['pruned']:
            for pruned in step_data['pruned'][:3]:
                st.markdown(f"""
                <div style="background: #FEE2E2; padding: 10px; border-radius: 6px; margin: 5px 0;
                            opacity: 0.7;">
                    "{pruned['text']}"<br>
                    <small>分数: {pruned['cumulative_score']:.4f}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("无剪枝")


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Beam Search 可视化</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 <b>Beam Search</b> 是一种启发式搜索算法，在每一步保留 K 个最优候选序列（beam），
    平衡搜索质量与计算开销。与贪心搜索（只保留 1 个）相比，能找到更优的全局解。
    </div>
    """, unsafe_allow_html=True)
    
    # 模型选择
    model_choice = st.selectbox(
        "选择模型",
        options=list(DEMO_MODELS.keys()),
        help="选择用于演示的模型"
    )
    
    model_info = DEMO_MODELS[model_choice]
    
    with st.spinner(f"加载 {model_choice}..."):
        model, tokenizer = load_model_and_tokenizer(model_info['id'])
    
    if model is None:
        st.error("模型加载失败")
        return
    
    st.success(f"✅ 模型已加载")
    
    st.markdown("---")
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        beam_size = st.slider(
            "Beam Size",
            min_value=2,
            max_value=5,
            value=3,
            help="每步保留的候选数量"
        )
    
    with col2:
        max_steps = st.slider(
            "最大步数",
            min_value=3,
            max_value=10,
            value=5,
            help="生成的最大 token 数"
        )
    
    with col3:
        st.metric("搜索空间", f"{beam_size}^{max_steps} = {beam_size**max_steps}")
    
    # Prompt 输入
    prompt = st.text_input(
        "输入 Prompt",
        value="Once upon a time",
        placeholder="输入起始文本..."
    )
    
    if st.button("开始 Beam Search", type="primary", width="stretch"):
        if not prompt:
            st.warning("请输入 Prompt")
            return
        
        with st.spinner("执行 Beam Search..."):
            history = beam_search_with_history(
                model, tokenizer, prompt,
                beam_size=beam_size,
                max_steps=max_steps
            )
        
        # 结果展示
        st.markdown("## 搜索结果")
        
        # 最终序列
        st.markdown("### 🏆 最终候选序列")
        
        for beam in history['final_beams']:
            rank_emoji = "🥇" if beam['rank'] == 1 else ("🥈" if beam['rank'] == 2 else "🥉")
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #DBEAFE, #F3F4F6); 
                        padding: 15px; border-radius: 8px; margin: 10px 0;">
                <span style="font-size: 20px;">{rank_emoji}</span>
                <b>Rank {beam['rank']}</b><br>
                <span style="font-family: monospace; font-size: 16px;">
                    {prompt}<b style="color: #2563EB;">{beam['sequence']}</b>
                </span><br>
                <small>累积 Log-Prob: {beam['score']:.4f} | 
                       概率: {beam['probability']:.2e}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # 搜索树可视化
        st.markdown("### 🌳 搜索树")
        fig = render_beam_tree(history)
        st.plotly_chart(fig, width='stretch')
        
        # 逐步详情
        st.markdown("### 📋 逐步详情")
        
        for step in history['steps']:
            with st.expander(f"Step {step['step'] + 1}", expanded=(step['step'] == 0)):
                render_step_detail(step)
        
        # 原理说明
        st.markdown("---")
        st.markdown("### 📚 Beam Search 原理")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            **算法流程**:
            1. 初始化 K 个 beam（初始只有 prompt）
            2. 对每个 beam，计算所有可能的下一个 token
            3. 从 K × V 个候选中选择 top-K（按累积概率）
            4. 重复直到达到最大长度或遇到 EOS
            
            **累积概率计算**:
            ```
            score(seq) = Σ log P(t_i | t_<i)
            ```
            使用 log 概率避免数值下溢。
            """)
        
        with col_b:
            st.markdown("""
            **对比其他策略**:
            
            | 策略 | Beam Size | 特点 |
            |------|-----------|------|
            | Greedy | 1 | 快但易陷入局部最优 |
            | Beam Search | K | 平衡质量与速度 |
            | Exhaustive | ∞ | 最优但计算量爆炸 |
            
            **Length Penalty**:
            实际应用中常加入长度惩罚：
            ```
            score = log_prob / length^α
            ```
            α > 1 偏好短序列，α < 1 偏好长序列
            """)

