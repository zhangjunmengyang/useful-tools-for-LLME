"""
TokenLab - LLM Tokenizer Visualization Workbench
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="TokenLab",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/tokenlab/tokenlab",
        "Report a bug": "https://github.com/tokenlab/tokenlab/issues",
        "About": "TokenLab - LLM Tokenizer Workbench"
    }
)

from utils.styles import GLOBAL_CSS

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
        padding: 4px 0 8px 20px !important;
        border: none !important;
        background: transparent !important;
    }
    
    /* ======= 导航按钮样式 ======= */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        padding: 8px 12px !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        background-color: transparent !important;
        color: #111827 !important;
        border: none !important;
        border-radius: 6px !important;
        margin: 2px 0 !important;
        box-shadow: none !important;
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
        "分词编码": "playground",
        "模型对比": "arena",
        "Chat Template": "chat_builder"
    },
    # 后续扩展示例：
    # "数据工具": {
    #     "清洗": "cleaner",
    #     "转换": "converter",
    # }
}

# 初始化
if "current_page" not in st.session_state:
    st.session_state.current_page = "分词编码"
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

# 获取当前模块
current_module = None
for group_name, items in NAV_STRUCTURE.items():
    if st.session_state.current_page in items:
        current_module = items[st.session_state.current_page]
        break

# 加载模块
if current_module == "playground":
    from pages import playground
    playground.render()
elif current_module == "arena":
    from pages import arena
    arena.render()
elif current_module == "chat_builder":
    from pages import chat_builder
    chat_builder.render()
