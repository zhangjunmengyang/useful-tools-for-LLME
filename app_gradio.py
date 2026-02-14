"""
LLM Tools Workbench - Gradio 版本
一个用于大模型学习与实验的工具集
"""

import gradio as gr

# 自定义 CSS 样式 - HuggingFace Style
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --color-accent: #FF9D00;
    --color-accent-light: #FFF7ED;
    --color-accent-dark: #EA580C;
    --color-success: #10B981;
    --color-warning: #F59E0B;
    --color-error: #EF4444;
    --color-info: #3B82F6;
    --font-mono: 'IBM Plex Mono', monospace;
    --hf-yellow: #FFD21E;
    --hf-orange: #FF9D00;
}

/* 全局样式 - HuggingFace Style */
.gradio-container {
    font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
}

/* HuggingFace 风格头部 */
.main-header {
    background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid #FED7AA;
}

/* 主要按钮 - HF 橙色 */
button.primary {
    background: linear-gradient(135deg, #FF9D00 0%, #EA580C 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #EA580C 0%, #C2410C 100%) !important;
}

/* 标题样式 */
.module-title {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    border-bottom: 2px solid #E5E7EB;
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}

/* Token 显示样式 */
.token-display {
    background-color: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px;
    line-height: 2.2;
    font-family: var(--font-mono);
}

.token {
    display: inline-block;
    padding: 4px 8px;
    margin: 2px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: default;
    transition: transform 0.15s ease;
}

.token:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.token-special {
    border: 2px solid #DC2626 !important;
}

.token-byte {
    border: 2px dashed #D97706 !important;
}

/* 统计卡片 */
.stat-card {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #FF9D00;
    font-family: var(--font-mono);
}

.stat-label {
    color: #64748B;
    font-size: 0.875rem;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-unit {
    color: #64748B;
    font-size: 0.875rem;
    font-weight: 500;
    margin-top: 4px;
}

/* Tab 样式增强 - HuggingFace Style */
.tab-nav button {
    font-weight: 600 !important;
    color: #6B7280 !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    color: #FF9D00 !important;
    background: #FFF7ED !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid #FF9D00 !important;
    color: #FF9D00 !important;
    background: #FFF7ED !important;
}

/* 信息面板 */
.info-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px;
}

/* 按钮样式 - HuggingFace Orange */
.primary-btn {
    background: linear-gradient(135deg, #FF9D00 0%, #EA580C 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    color: white !important;
}

.primary-btn:hover {
    background: linear-gradient(135deg, #EA580C 0%, #C2410C 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 157, 0, 0.3);
}

/* 预设按钮组 */
.preset-btn {
    background: #F3F4F6 !important;
    border: 1px solid #D1D5DB !important;
    color: #374151 !important;
    font-size: 0.875rem !important;
}

.preset-btn:hover {
    background: #E5E7EB !important;
}

/* Markdown 内容样式 */
.prose code {
    background: #F3F4F6;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--font-mono);
}

/* 提示框 - HuggingFace Style */
.tip-box {
    background: #FFF7ED;
    border-left: 4px solid #FF9D00;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
}

.warning-box {
    background: #FEF3C7;
    border-left: 4px solid #D97706;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
}

/* Dataframe 样式 */
.dataframe {
    font-size: 0.875rem !important;
}

/* 折叠面板 */
.accordion {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
}

/* 颜色变量 - Token 颜色 */
.token-color-0 { background-color: #D1FAE5; }
.token-color-1 { background-color: #DBEAFE; }
.token-color-2 { background-color: #E9D5FF; }
.token-color-3 { background-color: #FED7AA; }
.token-color-4 { background-color: #FBCFE8; }
.token-color-5 { background-color: #FEF08A; }
.token-color-6 { background-color: #CFFAFE; }
.token-color-7 { background-color: #FECDD3; }
.token-color-8 { background-color: #DDD6FE; }
.token-color-9 { background-color: #A7F3D0; }
.token-color-10 { background-color: #FFEDD5; }
.token-color-11 { background-color: #E2E8F0; }

/* 隐藏 Gradio 默认页脚 */
footer {
    display: none !important;
}

/* 修复按钮在 Row 中的背景问题 */
.row > .block {
    background: transparent !important;
}

/* 确保主要按钮样式正确 */
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #FF9D00 0%, #EA580C 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #EA580C 0%, #C2410C 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(255, 157, 0, 0.3) !important;
}

/* Group 容器样式优化 */
.group {
    background: transparent !important;
    border: none !important;
}

/* 表格样式优化 - 移除橙色边框 */
table {
    border-collapse: collapse !important;
}

table th, table td {
    border-color: #E5E7EB !important;
}

/* Markdown 表格样式 */
.prose table {
    border: 1px solid #E5E7EB !important;
}

.prose th, .prose td {
    border: 1px solid #E5E7EB !important;
    padding: 8px 12px !important;
}

/* Dataframe 样式优化 */
.dataframe {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* HTML 组件内的表格 */
.html-container table {
    border: none !important;
}

/* 输入框边框颜色 - 非聚焦状态 */
input, textarea, select {
    border-color: #D1D5DB !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #FF9D00 !important;
    box-shadow: 0 0 0 3px rgba(255, 157, 0, 0.1) !important;
}
"""

# Token 颜色列表
TOKEN_COLORS = [
    "#D1FAE5", "#DBEAFE", "#E9D5FF", "#FED7AA", "#FBCFE8", "#FEF08A",
    "#CFFAFE", "#FECDD3", "#DDD6FE", "#A7F3D0", "#FFEDD5", "#E2E8F0"
]

# 自定义主题 - HuggingFace Style
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="gray",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Source Sans Pro"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    radius_size="md",
    spacing_size="md",
).set(
    # 主要颜色
    button_primary_background_fill="linear-gradient(135deg, #FF9D00 0%, #EA580C 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #EA580C 0%, #C2410C 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",

    # 输入框聚焦颜色
    input_border_color_focus="#FF9D00",

    # 滑块颜色
    slider_color="#FF9D00",

    # 复选框颜色
    checkbox_background_color_selected="#FF9D00",
    checkbox_border_color_selected="#FF9D00",

    # 区块样式
    block_title_text_weight="600",
    block_label_text_weight="500",
)


def create_app():
    """创建 Gradio 应用"""

    # 收集所有需要初始化的 load 事件
    load_events = []

    with gr.Blocks(
        title="LLM Tools Workbench",
        analytics_enabled=False
    ) as app:

        # 标题 - HuggingFace Style Header
        gr.HTML("""
        <div style="
            background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 24px;
            border: 1px solid #FED7AA;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <div style="
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, #FF9D00 0%, #EA580C 100%);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
            ">
                <span style="filter: brightness(0) invert(1);">&#x1F9E0;</span>
            </div>
            <div>
                <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700; color: #1F2937;">
                    LLM Tools Workbench
                </h1>
                <p style="margin: 4px 0 0 0; color: #6B7280; font-size: 1rem;">
                    Interactive tools for exploring Large Language Models
                </p>
            </div>
        </div>
        """)
        
        # 主 Tab 导航
        with gr.Tabs() as main_tabs:
            
            # ==================== TokenLab ====================
            with gr.Tab("TokenLab", id="tokenlab"):
                from token_lab import playground, arena, chat_builder
                
                with gr.Tabs() as token_tabs:
                    with gr.Tab("Playground", id="playground"):
                        result = playground.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Arena", id="arena"):
                        result = arena.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Chat Template", id="chat"):
                        chat_builder.render()
            
            # ==================== EmbeddingLab ====================
            with gr.Tab("EmbeddingLab", id="embeddinglab"):
                from embedding_lab import (
                    vector_arithmetic,
                    model_comparison,
                    vector_visualization,
                    semantic_similarity
                )
                
                with gr.Tabs() as embed_tabs:
                    with gr.Tab("Vector Arithmetic", id="vector_arithmetic"):
                        result = vector_arithmetic.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Model Comparison", id="model_comparison"):
                        result = model_comparison.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Visualization", id="vector_viz"):
                        result = vector_visualization.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Semantic Similarity", id="semantic_sim"):
                        result = semantic_similarity.render()
                        if result:
                            load_events.append(result)
            
            # ==================== GenerationLab ====================
            with gr.Tab("GenerationLab", id="generationlab"):
                from generation_lab import (
                    logits_inspector,
                    beam_visualizer,
                    kv_cache_sim
                )
                
                with gr.Tabs() as gen_tabs:
                    with gr.Tab("Logits Inspector", id="logits"):
                        result = logits_inspector.render()
                        if result:
                            load_events.append(result)
                    
                    with gr.Tab("Beam Search", id="beam"):
                        result = beam_visualizer.render()
                        if result:
                            load_events.append(result)
                    
                    with gr.Tab("KV Cache", id="kvcache"):
                        result = kv_cache_sim.render()
                        if result:
                            load_events.append(result)
            
            # ==================== InterpretabilityLab ====================
            with gr.Tab("InterpretabilityLab", id="interpretabilitylab"):
                from interpretability_lab import (
                    attention_map,
                    rope_explorer,
                    ffn_activation
                )
                
                with gr.Tabs() as interp_tabs:
                    with gr.Tab("Attention", id="attention"):
                        result = attention_map.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("RoPE Explorer", id="rope"):
                        rope_explorer.render()

                    with gr.Tab("FFN Activation", id="ffn"):
                        result = ffn_activation.render()
                        if result:
                            load_events.append(result)
            
            # ==================== DataLab ====================
            with gr.Tab("DataLab", id="datalab"):
                from data_lab import (
                    hf_dataset_viewer,
                    cleaner_playground,
                    instruct_formatter
                )
                
                with gr.Tabs() as data_tabs:
                    with gr.Tab("Dataset Viewer", id="dataset"):
                        hf_dataset_viewer.render()

                    with gr.Tab("Data Cleaner", id="cleaner"):
                        cleaner_playground.render()

                    with gr.Tab("Format Converter", id="formatter"):
                        instruct_formatter.render()
            
            # ==================== ModelLab ====================
            with gr.Tab("ModelLab", id="modellab"):
                from model_lab import (
                    memory_estimator,
                    peft_calculator,
                    config_diff
                )
                
                with gr.Tabs() as model_tabs:
                    with gr.Tab("Memory Estimator", id="memory"):
                        result = memory_estimator.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("PEFT Calculator", id="peft"):
                        result = peft_calculator.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Config Diff", id="config"):
                        result = config_diff.render()
                        if result:
                            load_events.append(result)

            # ==================== RAGLab ====================
            with gr.Tab("RAGLab", id="raglab"):
                from rag_lab import (
                    chunking_playground,
                    retrieval_sim
                )

                with gr.Tabs() as rag_tabs:
                    with gr.Tab("Chunking", id="chunking"):
                        result = chunking_playground.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Retrieval", id="retrieval"):
                        result = retrieval_sim.render()
                        if result:
                            load_events.append(result)

            # ==================== FineTuneLab ====================
            with gr.Tab("FineTuneLab", id="finetunelab"):
                from finetune_lab import (
                    lora_explorer,
                    training_cost_estimator
                )

                with gr.Tabs() as finetune_tabs:
                    with gr.Tab("LoRA Explorer", id="lora_explorer"):
                        result = lora_explorer.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Training Cost", id="training_cost"):
                        result = training_cost_estimator.render()
                        if result:
                            load_events.append(result)

            # ==================== InferenceLab ====================
            with gr.Tab("InferenceLab", id="inferencelab"):
                from inference_lab import (
                    throughput_calculator,
                    quantization_analyzer
                )

                with gr.Tabs() as inference_tabs:
                    with gr.Tab("Throughput", id="throughput"):
                        result = throughput_calculator.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Quantization", id="quantization"):
                        result = quantization_analyzer.render()
                        if result:
                            load_events.append(result)

            # ==================== Agent Trace Lab ====================
            with gr.Tab("Agent Trace Lab", id="agenttracelab"):
                from agent_trace_lab import (
                    trace_viewer,
                    trace_analyzer
                )

                with gr.Tabs() as agent_trace_tabs:
                    with gr.Tab("Trace Viewer", id="trace_viewer"):
                        result = trace_viewer.render()
                        if result:
                            load_events.append(result)

                    with gr.Tab("Trace Analyzer", id="trace_analyzer"):
                        result = trace_analyzer.render()
                        if result:
                            load_events.append(result)


        # 页面加载时执行所有初始化函数
        if load_events:
            def combined_load():
                """合并所有 load 事件"""
                all_outputs = []
                for event in load_events:
                    try:
                        result = event['load_fn']()
                        if isinstance(result, tuple):
                            all_outputs.extend(result)
                        else:
                            all_outputs.append(result)
                    except Exception as e:
                        print(f"Load event error: {e}")
                        # 填充空值
                        all_outputs.extend([None] * len(event['load_outputs']))
                return all_outputs
            
            # 收集所有输出组件
            all_load_outputs = []
            for event in load_events:
                all_load_outputs.extend(event['load_outputs'])
            
            app.load(fn=combined_load, outputs=all_load_outputs)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS
    )
