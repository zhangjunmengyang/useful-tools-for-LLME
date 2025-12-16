"""
Beam Search 可视化 - 展示搜索路径树
"""

import gradio as gr
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
        } for p in pruned[:beam_size]]
        
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
    node_positions = {(0, 0): 0}
    
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
            
            # 添加边
            if step_idx == 0:
                parent_idx = 0
            else:
                parent_idx = node_positions.get((step_idx, 0), 0)
            
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
        height=450,
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


# 模型状态缓存
_loaded_model = {"name": None, "model": None, "tokenizer": None}


def run_beam_search(model_choice, prompt, beam_size, max_steps):
    """执行 Beam Search"""
    if not prompt:
        return "", None, ""
    
    model_info = DEMO_MODELS[model_choice]
    
    # 加载模型
    if _loaded_model["name"] != model_info['id']:
        model, tokenizer = load_model_and_tokenizer(model_info['id'])
        if model is None:
            return "模型加载失败", None, ""
        _loaded_model["name"] = model_info['id']
        _loaded_model["model"] = model
        _loaded_model["tokenizer"] = tokenizer
    else:
        model = _loaded_model["model"]
        tokenizer = _loaded_model["tokenizer"]
    
    # 执行搜索
    history = beam_search_with_history(
        model, tokenizer, prompt,
        beam_size=beam_size,
        max_steps=max_steps
    )
    
    # 最终序列展示
    results_md = "### 最终候选序列\n\n"
    for beam in history['final_beams']:
        results_md += f"""
**Rank {beam['rank']}**

`{prompt}`**{beam['sequence']}**

- 累积 Log-Prob: {beam['score']:.4f}
- 概率: {beam['probability']:.2e}

---
"""
    
    # 搜索树
    fig = render_beam_tree(history)
    
    # 逐步详情
    steps_md = "### 逐步详情\n\n"
    for step in history['steps']:
        steps_md += f"#### Step {step['step'] + 1}\n\n"
        steps_md += "**保留的 Beam:**\n\n"
        for i, beam in enumerate(step['active_beams']):
            steps_md += f"- Beam {i+1}: `{beam['text']}` (序列: {beam['full_sequence'][:50]}..., 分数: {beam['cumulative_score']:.4f})\n"
        
        if step['pruned']:
            steps_md += "\n**被剪枝:**\n\n"
            for pruned in step['pruned'][:3]:
                steps_md += f"- `{pruned['text']}` (分数: {pruned['cumulative_score']:.4f})\n"
        steps_md += "\n"
    
    return results_md, fig, steps_md


def render():
    """渲染页面"""
    
    gr.Markdown("## Beam Search 可视化")
    
    # 模型选择
    model_choice = gr.Dropdown(
        choices=list(DEMO_MODELS.keys()),
        value=list(DEMO_MODELS.keys())[0],
        label="选择模型"
    )
    
    # 参数设置
    with gr.Row():
        beam_size = gr.Slider(
            label="Beam Size",
            minimum=2,
            maximum=5,
            value=3,
            step=1
        )
        max_steps = gr.Slider(
            label="最大步数",
            minimum=3,
            maximum=10,
            value=5,
            step=1
        )
        search_space = gr.Textbox(
            label="搜索空间",
            interactive=False,
            value="3^5 = 243"
        )
    
    def update_search_space(beam, steps):
        return f"{beam}^{steps} = {beam**steps}"
    
    beam_size.change(fn=update_search_space, inputs=[beam_size, max_steps], outputs=[search_space])
    max_steps.change(fn=update_search_space, inputs=[beam_size, max_steps], outputs=[search_space])
    
    # Prompt 输入
    prompt = gr.Textbox(
        label="Prompt",
        value="Once upon a time",
        placeholder="输入起始文本..."
    )
    
    # 结果展示
    results_md = gr.Markdown("")
    tree_plot = gr.Plot(label="搜索树")
    
    with gr.Accordion("逐步详情", open=False):
        steps_detail = gr.Markdown("")
    
    # 参数变化自动触发搜索
    for component in [model_choice, prompt, beam_size, max_steps]:
        component.change(
            fn=run_beam_search,
            inputs=[model_choice, prompt, beam_size, max_steps],
            outputs=[results_md, tree_plot, steps_detail]
        )
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        return run_beam_search(list(DEMO_MODELS.keys())[0], "Once upon a time", 3, 5)
    
    # 返回 load 事件信息
    return {
        'load_fn': on_load,
        'load_outputs': [results_md, tree_plot, steps_detail]
    }
