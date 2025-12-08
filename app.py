"""
LLM Tools Workbench - LLM 工具集
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="LLM Tools Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/tokenlab/tokenlab",
        "Report a bug": "https://github.com/tokenlab/tokenlab/issues",
        "About": "LLM Tools Workbench - 大模型学习与实验平台"
    }
)

from shared.styles import GLOBAL_CSS

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# 侧边栏导航样式
SIDEBAR_NAV_CSS = """
<style>
    /* 隐藏 Streamlit 默认的页面导航 */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* 侧边栏标题 */
    .sidebar-header {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0 0 12px 0;
    }
    
    /* ======= Expander 无边框样式 ======= */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        margin: 0 0 4px 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
    }
    
    /* Summary 标题样式 */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #111827 !important;
        padding: 8px 12px !important;
        border: none !important;
        background: transparent !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background-color: #F3F4F6 !important;
    }
    
    /* 展开内容区域 */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 4px 0 8px 0 !important;
        border: none !important;
        background: transparent !important;
        margin-left: 12px !important;
        border-left: 1px solid #E5E7EB !important;
        padding-left: 12px !important;
    }
    
    /* ======= 导航按钮样式 ======= */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 6px 12px !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        background-color: transparent !important;
        color: #4B5563 !important;
        border: none !important;
        border-radius: 6px !important;
        margin: 1px 0 !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .stButton > button > div {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    [data-testid="stSidebar"] .stButton > button > div > p {
        text-align: left !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #F3F4F6 !important;
        color: #111827 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* 选中状态 */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #DBEAFE !important;
        color: #2563EB !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #BFDBFE !important;
        color: #2563EB !important;
    }
</style>
"""

st.markdown(SIDEBAR_NAV_CSS, unsafe_allow_html=True)

# 多级导航结构
NAV_STRUCTURE = {
    "TokenLab": {
        "Tokenizer Playground": "playground",
        "Tokenizer Arena": "arena",
        "Chat Template": "chat_builder"
    },
    "EmbeddingLab": {
        "Vector Arithmetic": "vector_arithmetic",
        "Model Comparison": "embedding_comparison",
        "Vector Visualization": "vector_visualization",
        "Semantic Similarity": "semantic_similarity"
    },
    "GenerationLab": {
        "Logits Inspector": "logits_inspector",
        "Beam Search Visualizer": "beam_visualizer",
        "KV Cache Simulator": "kv_cache_sim"
    },
    "InterpretabilityLab": {
        "Attention Visualizer": "attention_map",
        "RoPE Explorer": "rope_explorer",
        "FFN Analyzer": "ffn_activation"
    },
    "DataLab": {
        "Dataset Viewer": "hf_dataset_viewer",
        "Data Cleaner": "cleaner_playground",
        "Instruct Formatter": "instruct_formatter"
    },
    "ModelLab": {
        "Memory Estimator": "memory_estimator",
        "PEFT Calculator": "peft_calculator",
        "Config Diff": "config_diff"
    }
}

# 初始化
if "current_page" not in st.session_state:
    st.session_state.current_page = "Tokenizer Playground"
if "current_group" not in st.session_state:
    st.session_state.current_group = "TokenLab"

# 侧边栏
with st.sidebar:
    st.markdown('<div class="sidebar-header">Tools</div>', unsafe_allow_html=True)
    
    for group_name, items in NAV_STRUCTURE.items():
        is_group_active = st.session_state.current_group == group_name
        
        with st.expander(group_name, expanded=is_group_active):
            for item_name, module_name in items.items():
                is_active = (st.session_state.current_page == item_name and 
                           st.session_state.current_group == group_name)
                
                if st.button(
                    item_name,
                    key=f"nav_{group_name}_{item_name}",
                    width="stretch",
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_page = item_name
                    st.session_state.current_group = group_name
                    st.rerun()

# 获取当前模块（需要同时匹配 group 和 page）
current_module = None
current_group = st.session_state.current_group
current_page = st.session_state.current_page
if current_group in NAV_STRUCTURE and current_page in NAV_STRUCTURE[current_group]:
    current_module = NAV_STRUCTURE[current_group][current_page]

# 加载模块
# TokenLab 模块
if current_module == "playground":
    from token_lab import playground
    playground.render()
elif current_module == "arena":
    from token_lab import arena
    arena.render()
elif current_module == "chat_builder":
    from token_lab import chat_builder
    chat_builder.render()
# EmbeddingLab 模块
elif current_module == "vector_arithmetic":
    from embedding_lab import vector_arithmetic
    vector_arithmetic.render()
elif current_module == "embedding_comparison":
    from embedding_lab import model_comparison
    model_comparison.render()
elif current_module == "vector_visualization":
    from embedding_lab import vector_visualization
    vector_visualization.render()
elif current_module == "semantic_similarity":
    from embedding_lab import semantic_similarity
    semantic_similarity.render()
# GenerationLab 模块
elif current_module == "logits_inspector":
    from generation_lab import logits_inspector
    logits_inspector.render()
elif current_module == "beam_visualizer":
    from generation_lab import beam_visualizer
    beam_visualizer.render()
elif current_module == "kv_cache_sim":
    from generation_lab import kv_cache_sim
    kv_cache_sim.render()
# InterpretabilityLab 模块
elif current_module == "attention_map":
    from interpretability_lab import attention_map
    attention_map.render()
elif current_module == "rope_explorer":
    from interpretability_lab import rope_explorer
    rope_explorer.render()
elif current_module == "ffn_activation":
    from interpretability_lab import ffn_activation
    ffn_activation.render()
# DataLab 模块
elif current_module == "hf_dataset_viewer":
    from data_lab import hf_dataset_viewer
    hf_dataset_viewer.render()
elif current_module == "cleaner_playground":
    from data_lab import cleaner_playground
    cleaner_playground.render()
elif current_module == "instruct_formatter":
    from data_lab import instruct_formatter
    instruct_formatter.render()
# ModelLab 模块
elif current_module == "memory_estimator":
    from model_lab import memory_estimator
    memory_estimator.render()
elif current_module == "peft_calculator":
    from model_lab import peft_calculator
    peft_calculator.render()
elif current_module == "config_diff":
    from model_lab import config_diff
    config_diff.render()
