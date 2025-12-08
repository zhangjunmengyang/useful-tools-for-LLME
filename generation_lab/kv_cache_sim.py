"""
KV Cache 模拟器 - 可视化推理过程中的显存占用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from generation_lab.generation_utils import (
    MODEL_CONFIGS,
    calculate_kv_cache_size,
    simulate_kv_cache_growth,
    simulate_paged_attention,
    format_bytes
)


def render_kv_cache_growth_chart(growth_data: list) -> go.Figure:
    """渲染 KV Cache 增长曲线"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("KV Cache 累积显存", "每步增量"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    steps = [d['step'] for d in growth_data]
    cache_gb = [d['cache_gb'] for d in growth_data]
    delta_gb = [d['delta_gb'] * 1000 for d in growth_data]  # 转为 MB
    phases = [d['phase'] for d in growth_data]
    
    # 颜色映射
    colors = ['#DC2626' if p == 'Prefill' else '#2563EB' for p in phases]
    
    # 累积曲线
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=cache_gb,
            mode='lines+markers',
            name='累积显存',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=8, color=colors),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        ),
        row=1, col=1
    )
    
    # 标注 Prefill 点
    prefill_idx = 0
    fig.add_annotation(
        x=steps[prefill_idx],
        y=cache_gb[prefill_idx],
        text=f"Prefill<br>{cache_gb[prefill_idx]:.3f} GB",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color='#DC2626'),
        row=1, col=1
    )
    
    # 增量柱状图
    fig.add_trace(
        go.Bar(
            x=steps,
            y=delta_gb,
            name='每步增量',
            marker_color=colors,
            text=[f'{d:.2f}' for d in delta_gb],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40)
    )
    
    fig.update_yaxes(title_text="显存 (GB)", row=1, col=1)
    fig.update_yaxes(title_text="增量 (MB)", row=2, col=1)
    fig.update_xaxes(title_text="生成步骤", row=2, col=1)
    
    return fig


def render_paged_attention_viz(paged_data: dict) -> go.Figure:
    """渲染 PagedAttention Block 分配图"""
    sequences = paged_data['sequences']
    block_size = paged_data['block_size']
    
    # 创建 block 网格
    total_blocks = paged_data['total_blocks']
    cols = 16
    rows = (total_blocks + cols - 1) // cols
    
    # 初始化网格
    grid = np.zeros((rows, cols))
    annotations = []
    
    colors = px.colors.qualitative.Set2
    block_idx = 0
    
    for seq in sequences:
        seq_color_idx = seq['seq_id'] % len(colors)
        for _ in range(seq['blocks']):
            if block_idx < total_blocks:
                row = block_idx // cols
                col = block_idx % cols
                grid[row, col] = seq['seq_id'] + 1
                block_idx += 1
    
    # 创建热力图
    fig = go.Figure()
    
    # 自定义颜色映射
    colorscale = [
        [0, '#E5E7EB'],  # 空闲
    ]
    for i in range(len(sequences)):
        val = (i + 1) / (len(sequences) + 1)
        colorscale.append([val, colors[i % len(colors)]])
    colorscale.append([1, colors[-1]])
    
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=colorscale,
        showscale=False,
        hovertemplate='Block %{x},%{y}<br>Sequence: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"PagedAttention Block 分配 (Block Size = {block_size})",
        xaxis=dict(title="Column", showgrid=False),
        yaxis=dict(title="Row", showgrid=False, autorange='reversed'),
        height=300,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig


def render_utilization_chart(paged_data: dict) -> go.Figure:
    """渲染利用率图表"""
    sequences = paged_data['sequences']
    
    fig = go.Figure()
    
    seq_ids = [f"Seq {s['seq_id']}" for s in sequences]
    utilizations = [s['utilization'] for s in sequences]
    
    fig.add_trace(go.Bar(
        x=seq_ids,
        y=utilizations,
        marker_color=['#2563EB' if u > 80 else '#D97706' if u > 50 else '#DC2626' 
                      for u in utilizations],
        text=[f'{u:.1f}%' for u in utilizations],
        textposition='outside'
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="#059669",
                  annotation_text="理想利用率")
    
    fig.update_layout(
        title="各序列 Block 利用率",
        xaxis_title="序列",
        yaxis_title="利用率 (%)",
        yaxis_range=[0, 110],
        height=300
    )
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">KV Cache 模拟器</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 <b>KV Cache</b> 是 Transformer 推理加速的关键技术。在自回归生成中，
    缓存已计算的 Key/Value 避免重复计算，但会占用大量显存。
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 显存计算", "📈 增长模拟", "🧩 PagedAttention"])
    
    with tab1:
        st.markdown("### KV Cache 显存计算器")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 模型选择
            model_choice = st.selectbox(
                "选择预设模型",
                options=["自定义"] + list(MODEL_CONFIGS.keys())
            )
            
            if model_choice == "自定义":
                num_layers = st.number_input("层数", value=32, min_value=1, max_value=200)
                hidden_size = st.number_input("Hidden Size", value=4096, min_value=64, max_value=32768)
                num_heads = st.number_input("注意力头数", value=32, min_value=1, max_value=128)
            else:
                config = MODEL_CONFIGS[model_choice]
                num_layers = config['num_hidden_layers']
                hidden_size = config['hidden_size']
                num_heads = config['num_attention_heads']
                
                st.info(f"""
                **{model_choice}** 配置:
                - Layers: {num_layers}
                - Hidden: {hidden_size}
                - Heads: {num_heads}
                """)
            
            st.markdown("---")
            
            seq_length = st.number_input("序列长度", value=2048, min_value=1, max_value=131072)
            batch_size = st.number_input("Batch Size", value=1, min_value=1, max_value=256)
            dtype = st.selectbox("数据类型", ["float16/bfloat16", "float32", "int8"])
            
            dtype_bytes = {"float16/bfloat16": 2, "float32": 4, "int8": 1}[dtype]
        
        with col2:
            # 计算结果
            result = calculate_kv_cache_size(
                num_layers, hidden_size, num_heads,
                seq_length, batch_size, dtype_bytes
            )
            
            st.markdown("### 计算结果")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("总 KV Cache", f"{result['total_gb']:.3f} GB")
            with metric_cols[1]:
                st.metric("每层占用", f"{result['per_layer_mb']:.2f} MB")
            with metric_cols[2]:
                st.metric("K Cache", format_bytes(result['k_cache_bytes']))
            with metric_cols[3]:
                st.metric("V Cache", format_bytes(result['v_cache_bytes']))
            
            st.markdown("---")
            
            # 公式说明
            st.markdown("""
            #### 📐 计算公式
            
            ```
            KV Cache = 2 × num_layers × batch_size × seq_length × hidden_size × dtype_bytes
            
            其中:
            - 2: K 和 V 各一份
            - hidden_size = num_heads × head_dim
            ```
            """)
            
            # 详细分解
            st.markdown("#### 详细分解")
            
            breakdown_data = {
                "组件": ["K Cache", "V Cache", "单层 KV", "总计"],
                "公式": [
                    f"{num_layers} × {batch_size} × {seq_length} × {hidden_size} × {dtype_bytes}",
                    f"{num_layers} × {batch_size} × {seq_length} × {hidden_size} × {dtype_bytes}",
                    f"2 × {batch_size} × {seq_length} × {hidden_size} × {dtype_bytes}",
                    f"2 × {num_layers} × {batch_size} × {seq_length} × {hidden_size} × {dtype_bytes}"
                ],
                "大小": [
                    format_bytes(result['k_cache_bytes']),
                    format_bytes(result['v_cache_bytes']),
                    format_bytes(result['per_layer_bytes']),
                    format_bytes(result['total_bytes'])
                ]
            }
            
            st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, width="stretch")
    
    with tab2:
        st.markdown("### Prefill vs Decode 阶段模拟")
        
        st.markdown("""
        Transformer 推理分为两个阶段：
        - **Prefill**: 一次性处理整个 Prompt，计算所有位置的 KV Cache
        - **Decode**: 逐 token 生成，每步只计算一个新位置的 KV
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sim_model = st.selectbox(
                "选择模型",
                options=list(MODEL_CONFIGS.keys()),
                key="sim_model"
            )
        
        with col2:
            prompt_len = st.number_input("Prompt 长度", value=512, min_value=1, max_value=8192)
        
        with col3:
            gen_len = st.number_input("生成长度", value=128, min_value=1, max_value=2048)
        
        if st.button("开始模拟", type="primary"):
            config = MODEL_CONFIGS[sim_model]
            
            growth_data = simulate_kv_cache_growth(
                config, prompt_len, gen_len,
                batch_size=1, dtype_bytes=2
            )
            
            # 显示关键指标
            final_cache = growth_data[-1]['cache_gb']
            prefill_cache = growth_data[0]['cache_gb']
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Prefill 后", f"{prefill_cache:.3f} GB")
            with col_b:
                st.metric("最终显存", f"{final_cache:.3f} GB")
            with col_c:
                st.metric("Decode 增量", f"{(final_cache - prefill_cache) * 1000:.1f} MB")
            
            # 增长曲线
            fig = render_kv_cache_growth_chart(growth_data)
            st.plotly_chart(fig, width='stretch')
            
            # 数据表
            with st.expander("详细数据"):
                df = pd.DataFrame([{
                    "步骤": d['step'],
                    "阶段": d['phase'],
                    "序列长度": d['seq_length'],
                    "累积显存 (GB)": f"{d['cache_gb']:.4f}",
                    "增量 (MB)": f"{d['delta_gb'] * 1000:.2f}",
                    "说明": d['description']
                } for d in growth_data])
                st.dataframe(df, hide_index=True, width="stretch")
    
    with tab3:
        st.markdown("### PagedAttention 模拟")
        
        st.markdown("""
        **PagedAttention** (vLLM) 将 KV Cache 分割成固定大小的 Block，类似操作系统的内存分页：
        - 避免为最大序列长度预分配显存
        - 支持多序列动态调度
        - 缺点：产生内部碎片
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            block_size = st.selectbox("Block Size", [8, 16, 32, 64], index=1)
        
        with col2:
            num_seqs = st.slider("并发序列数", 2, 8, 4)
        
        with col3:
            avg_tokens = st.number_input("平均 Token 数", value=256, min_value=32)
        
        if st.button("模拟分配", key="paged_sim"):
            paged_data = simulate_paged_attention(avg_tokens, block_size, num_seqs)
            
            # 统计指标
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("总 Block 数", paged_data['total_blocks'])
            with col_b:
                st.metric("总容量", f"{paged_data['total_capacity']} tokens")
            with col_c:
                st.metric("实际使用", f"{paged_data['total_tokens']} tokens")
            with col_d:
                st.metric("整体利用率", f"{paged_data['overall_utilization']:.1f}%")
            
            # Block 分配可视化
            st.markdown("#### Block 分配图")
            st.caption("每种颜色代表一个序列")
            
            fig_blocks = render_paged_attention_viz(paged_data)
            st.plotly_chart(fig_blocks, width='stretch')
            
            # 利用率分析
            fig_util = render_utilization_chart(paged_data)
            st.plotly_chart(fig_util, width='stretch')
            
            # 序列详情
            st.markdown("#### 各序列详情")
            seq_df = pd.DataFrame([{
                "序列 ID": s['seq_id'],
                "Token 数": s['length'],
                "Block 数": s['blocks'],
                "浪费 (tokens)": s['waste'],
                "利用率": f"{s['utilization']:.1f}%"
            } for s in paged_data['sequences']])
            st.dataframe(seq_df, hide_index=True, width="stretch")
            
            # 碎片分析
            st.markdown("#### 内部碎片分析")
            st.markdown(f"""
            - **Block Size**: {block_size} tokens
            - **总浪费**: {paged_data['total_waste']} tokens ({100 - paged_data['overall_utilization']:.1f}%)
            - **建议**: Block Size 越小，碎片越少，但管理开销越大
            """)
    
    # 原理说明
    st.markdown("---")
    st.markdown("### 📚 KV Cache 原理详解")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        #### 为什么需要 KV Cache？
        
        在自回归生成中，每个新 token 需要 attend 到之前所有 token：
        
        ```
        Attention(Q, K, V) = softmax(Q K^T / √d) V
        ```
        
        - 无缓存：每次重新计算所有位置的 K, V
        - 有缓存：只计算新位置，复用之前的 K, V
        
        **时间复杂度**：
        - 无缓存: O(n² × d) 每步
        - 有缓存: O(n × d) 每步
        """)
    
    with col_right:
        st.markdown("""
        #### GQA 如何减少 KV Cache？
        
        **Grouped Query Attention (GQA)** 多个 Query Head 共享一组 KV：
        
        | 模型 | KV Heads | 相比 MHA 节省 |
        |------|----------|---------------|
        | Llama-2-7B | 32 (MHA) | 0% |
        | Llama-3-8B | 8 (GQA) | 75% |
        | Llama-2-70B | 8 (GQA) | 87.5% |
        
        **GQA Cache 公式**:
        ```
        KV Cache = 2 × L × B × S × (H_kv × d_head) × dtype
        ```
        其中 H_kv 是 KV head 数量（可能远小于 Q head 数量）
        """)

