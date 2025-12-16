"""
LLM Tools Workbench - Gradio 版本
一个用于大模型学习与实验的工具集
"""

import gradio as gr

# 自定义 CSS 样式
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --color-accent: #2563EB;
    --color-accent-light: #DBEAFE;
    --color-success: #059669;
    --color-warning: #D97706;
    --color-error: #DC2626;
    --font-mono: 'JetBrains Mono', monospace;
}

/* 全局样式 */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
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
    font-weight: 600;
    color: var(--color-accent);
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

/* Tab 样式增强 */
.tab-nav button {
    font-weight: 500 !important;
}

.tab-nav button.selected {
    border-bottom: 2px solid var(--color-accent) !important;
}

/* 信息面板 */
.info-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px;
}

/* 按钮样式 */
.primary-btn {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    border: none !important;
    font-weight: 500 !important;
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

/* 提示框 */
.tip-box {
    background: #DBEAFE;
    border-left: 4px solid #2563EB;
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
"""

# Token 颜色列表
TOKEN_COLORS = [
    "#D1FAE5", "#DBEAFE", "#E9D5FF", "#FED7AA", "#FBCFE8", "#FEF08A",
    "#CFFAFE", "#FECDD3", "#DDD6FE", "#A7F3D0", "#FFEDD5", "#E2E8F0"
]

# 自定义主题
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


def create_app():
    """创建 Gradio 应用"""
    
    with gr.Blocks(title="LLM Tools Workbench") as app:
        
        # 标题
        gr.Markdown("""
        # LLM Tools Workbench
        <p style="color: #6B7280; margin-top: -8px;">大模型学习与实验平台 | Learning & Experimentation Platform for LLMs</p>
        """)
        
        # 主 Tab 导航
        with gr.Tabs() as main_tabs:
            
            # ==================== TokenLab ====================
            with gr.Tab("TokenLab", id="tokenlab"):
                from token_lab import playground, arena, chat_builder
                
                with gr.Tabs() as token_tabs:
                    with gr.Tab("分词编码", id="playground"):
                        playground.render()
                    
                    with gr.Tab("模型对比", id="arena"):
                        arena.render()
                    
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
                    with gr.Tab("向量运算", id="vector_arithmetic"):
                        vector_arithmetic.render()
                    
                    with gr.Tab("模型对比", id="model_comparison"):
                        model_comparison.render()
                    
                    with gr.Tab("向量可视化", id="vector_viz"):
                        vector_visualization.render()
                    
                    with gr.Tab("语义相似度", id="semantic_sim"):
                        semantic_similarity.render()
            
            # ==================== GenerationLab ====================
            with gr.Tab("GenerationLab", id="generationlab"):
                from generation_lab import (
                    logits_inspector,
                    beam_visualizer,
                    kv_cache_sim
                )
                
                with gr.Tabs() as gen_tabs:
                    with gr.Tab("Logits Inspector", id="logits"):
                        logits_inspector.render()
                    
                    with gr.Tab("Beam Search", id="beam"):
                        beam_visualizer.render()
                    
                    with gr.Tab("KV Cache", id="kvcache"):
                        kv_cache_sim.render()
            
            # ==================== InterpretabilityLab ====================
            with gr.Tab("InterpretabilityLab", id="interpretabilitylab"):
                from interpretability_lab import (
                    attention_map,
                    rope_explorer,
                    ffn_activation
                )
                
                with gr.Tabs() as interp_tabs:
                    with gr.Tab("Attention", id="attention"):
                        attention_map.render()
                    
                    with gr.Tab("RoPE 探索", id="rope"):
                        rope_explorer.render()
                    
                    with gr.Tab("FFN 激活", id="ffn"):
                        ffn_activation.render()
            
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
                    
                    with gr.Tab("数据清洗", id="cleaner"):
                        cleaner_playground.render()
                    
                    with gr.Tab("格式转换", id="formatter"):
                        instruct_formatter.render()
            
            # ==================== ModelLab ====================
            with gr.Tab("ModelLab", id="modellab"):
                from model_lab import (
                    memory_estimator,
                    peft_calculator,
                    config_diff
                )
                
                with gr.Tabs() as model_tabs:
                    with gr.Tab("显存估算", id="memory"):
                        memory_estimator.render()
                    
                    with gr.Tab("PEFT 计算器", id="peft"):
                        peft_calculator.render()
                    
                    with gr.Tab("Config 对比", id="config"):
                        config_diff.render()
        
        # 页脚
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #9CA3AF; font-size: 0.875rem;">
            LLM Tools Workbench | Built with Gradio
        </div>
        """)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=CUSTOM_THEME
    )
