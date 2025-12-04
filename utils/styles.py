"""
TokenLab - 样式配置
简约清爽的现代设计系统
"""

# =================================
# 配色系统 (基于 design.md 规范)
# =================================

COLORS = {
    # 中性色
    "white": "#FFFFFF",
    "gray_50": "#FAFBFC",
    "gray_100": "#F3F4F6",
    "gray_200": "#E5E7EB",
    "gray_300": "#D1D5DB",
    "gray_400": "#9CA3AF",
    "gray_500": "#6B7280",
    "gray_600": "#4B5563",
    "gray_900": "#111827",
    
    # 语义别名
    "bg_primary": "#FFFFFF",
    "bg_secondary": "#FAFBFC",
    "bg_tertiary": "#F3F4F6",
    "text_primary": "#111827",
    "text_secondary": "#4B5563",
    "text_muted": "#6B7280",
    "text_placeholder": "#9CA3AF",
    "border": "#D1D5DB",
    "border_light": "#E5E7EB",
    
    # 强调色
    "accent_blue": "#2563EB",
    "accent_blue_light": "#DBEAFE",
    "accent_blue_dark": "#1D4ED8",
    "accent_green": "#059669",
    "accent_green_light": "#D1FAE5",
    "accent_yellow": "#D97706",
    "accent_yellow_light": "#FEF3C7",
    "accent_red": "#DC2626",
    "accent_red_light": "#FEE2E2",
    "accent_purple": "#7C3AED",
}

# Token 显示颜色 (Pastel 系，确保深色文字可读)
TOKEN_COLORS = [
    "#D1FAE5",  # 薄荷绿
    "#DBEAFE",  # 天空蓝
    "#E9D5FF",  # 薰衣紫
    "#FED7AA",  # 蜜桃橙
    "#FBCFE8",  # 玫瑰粉
    "#FEF08A",  # 柠檬黄
    "#CFFAFE",  # 淡青色
    "#FECDD3",  # 桃粉色
    "#DDD6FE",  # 淡紫色
    "#A7F3D0",  # 薄荷青
    "#FFEDD5",  # 杏色
    "#E2E8F0",  # 石板灰
]

# =================================
# 全局 CSS 样式
# =================================

GLOBAL_CSS = """
<style>
    /* =================================
       字体引入
       ================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* =================================
       CSS 变量定义
       ================================= */
    :root {
        /* 字号系统 (1.25 倍模块缩放) */
        --font-size-micro: 11px;
        --font-size-caption: 12px;
        --font-size-sm: 13px;
        --font-size-base: 14px;
        --font-size-lg: 15px;
        --font-size-h3: 16px;
        --font-size-h2: 20px;
        --font-size-h1: 24px;
        --font-size-display: 28px;
        
        /* 行高 */
        --line-height-tight: 1.3;
        --line-height-snug: 1.4;
        --line-height-normal: 1.5;
        --line-height-relaxed: 1.6;
        
        /* 字重 */
        --font-weight-regular: 400;
        --font-weight-medium: 500;
        --font-weight-semibold: 600;
        
        /* 配色 */
        --color-bg-primary: #FFFFFF;
        --color-bg-secondary: #FAFBFC;
        --color-bg-tertiary: #F3F4F6;
        --color-text-primary: #111827;
        --color-text-secondary: #4B5563;
        --color-text-muted: #6B7280;
        --color-text-placeholder: #9CA3AF;
        --color-border: #D1D5DB;
        --color-border-light: #E5E7EB;
        --color-accent: #2563EB;
        --color-accent-light: #DBEAFE;
        --color-accent-dark: #1D4ED8;
        --color-success: #059669;
        --color-success-light: #D1FAE5;
        --color-warning: #D97706;
        --color-warning-light: #FEF3C7;
        --color-error: #DC2626;
        --color-error-light: #FEE2E2;
        
        /* 间距 */
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 12px;
        --spacing-lg: 16px;
        --spacing-xl: 24px;
        --spacing-2xl: 32px;
        
        /* 圆角 */
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
        
        /* 过渡 */
        --transition-fast: 0.15s ease;
        --transition-normal: 0.2s ease;
    }
    
    /* =================================
       全局基础样式
       ================================= */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--color-text-primary);
        font-size: var(--font-size-base);
        line-height: var(--line-height-relaxed);
    }
    
    .stApp {
        background-color: var(--color-bg-primary);
    }
    
    /* =================================
       标题样式
       ================================= */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: var(--font-weight-semibold);
        font-size: var(--font-size-h1) !important;
        color: var(--color-text-primary) !important;
        letter-spacing: -0.02em;
        line-height: var(--line-height-snug);
        margin-bottom: var(--spacing-md);
    }
    
    h2 {
        font-family: 'Inter', sans-serif;
        font-weight: var(--font-weight-semibold);
        font-size: var(--font-size-h2) !important;
        color: var(--color-text-primary) !important;
        line-height: var(--line-height-snug);
        margin-top: var(--spacing-xl);
        margin-bottom: var(--spacing-md);
    }
    
    h3, h4, h5 {
        font-family: 'Inter', sans-serif;
        font-weight: var(--font-weight-medium);
        font-size: var(--font-size-h3) !important;
        color: var(--color-text-primary) !important;
        line-height: var(--line-height-snug);
    }
    
    /* 模块标题样式 (Display 级) */
    .module-title {
        font-size: var(--font-size-display) !important;
        font-weight: var(--font-weight-semibold);
        color: var(--color-text-primary) !important;
        border-bottom: 1px solid var(--color-border-light);
        padding-bottom: var(--spacing-md);
        margin-bottom: var(--spacing-xl);
        letter-spacing: -0.02em;
    }
    
    /* =================================
       正文样式
       ================================= */
    p, span, div, label {
        color: var(--color-text-primary);
        font-size: var(--font-size-base);
    }
    
    /* 帮助图标 */
    [data-testid="stTooltipIcon"] {
        color: var(--color-text-muted);
        margin-left: var(--spacing-sm);
        vertical-align: middle;
    }
    [data-testid="stTooltipIcon"]:hover {
        color: var(--color-accent);
    }

    /* =================================
       输入框与交互组件
       ================================= */
    
    /* 文本输入框 */
    .stTextArea textarea, .stTextInput input {
        background-color: var(--color-bg-primary) !important;
        color: var(--color-text-primary) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: var(--font-size-base) !important;
        padding: var(--spacing-md) !important;
        line-height: var(--line-height-relaxed) !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--color-accent) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: var(--color-text-placeholder) !important;
        font-size: var(--font-size-sm) !important;
    }
    
    /* =================================
       下拉选择框
       ================================= */
    
    /* 选择框主体 */
    div[data-baseweb="select"] > div {
        background-color: var(--color-bg-primary) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        min-height: 40px !important;
        transition: all var(--transition-fast) !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: var(--color-accent) !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* 选择框内文字 */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: var(--color-text-primary) !important;
        font-size: var(--font-size-base) !important;
    }
    
    /* 模型选择器图标样式 */
    div[data-baseweb="select"] span[data-baseweb="tag"] {
        font-size: var(--font-size-lg) !important;
    }
    
    /* 下拉菜单容器 */
    div[data-baseweb="popover"] {
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        overflow: hidden !important;
    }
    
    /* 下拉菜单列表 */
    ul[data-baseweb="menu"],
    div[data-baseweb="menu"] {
        background-color: var(--color-bg-primary) !important;
        padding: var(--spacing-xs) !important;
    }
    
    /* 下拉选项 */
    li[data-baseweb="menu-item"],
    div[role="option"] {
        color: var(--color-text-primary) !important;
        background-color: var(--color-bg-primary) !important;
        border-radius: var(--radius-sm) !important;
        padding: var(--spacing-sm) var(--spacing-md) !important;
        margin: 0 !important;
        border: none !important;
        outline: none !important;
        font-size: var(--font-size-base) !important;
    }
    
    /* 下拉选项 hover */
    li[data-baseweb="menu-item"]:hover,
    div[role="option"]:hover,
    li[data-baseweb="menu-item"][aria-selected="true"],
    div[role="option"][aria-selected="true"] {
        background-color: var(--color-bg-tertiary) !important;
        color: var(--color-text-primary) !important;
        border: none !important;
        outline: none !important;
    }
    
    /* 移除聚焦边框 */
    li[data-baseweb="menu-item"]:focus,
    div[role="option"]:focus,
    li[data-baseweb="menu-item"]:focus-visible,
    div[role="option"]:focus-visible {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* 单选/复选框 */
    .stRadio label, .stCheckbox label {
        color: var(--color-text-primary) !important;
        font-size: var(--font-size-base) !important;
    }
    
    /* =================================
       按钮样式
       ================================= */
    
    /* 普通按钮 */
    .stApp > div > div > div > div > .stButton > button {
        background-color: var(--color-bg-tertiary) !important;
        color: var(--color-text-primary) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        font-weight: var(--font-weight-medium) !important;
        font-size: var(--font-size-base) !important;
        padding: var(--spacing-sm) var(--spacing-lg) !important;
        transition: all var(--transition-fast) !important;
    }
    
    .stApp > div > div > div > div > .stButton > button:hover {
        background-color: var(--color-border-light) !important;
        border-color: var(--color-text-muted) !important;
    }
    
    /* =================================
       Tab 标签页
       ================================= */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--color-border-light);
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--color-text-muted);
        background-color: transparent;
        border: none;
        padding: var(--spacing-md) var(--spacing-lg);
        font-size: var(--font-size-base);
        font-weight: var(--font-weight-medium);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--color-text-primary) !important;
        border-bottom: 2px solid var(--color-accent) !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--color-text-primary);
    }
    
    /* =================================
       Expander
       ================================= */
    .streamlit-expanderHeader {
        background-color: var(--color-bg-tertiary) !important;
        border: 1px solid var(--color-border-light) !important;
        border-radius: var(--radius-md) !important;
        color: var(--color-text-primary) !important;
        font-size: var(--font-size-base) !important;
        font-weight: var(--font-weight-medium) !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid var(--color-border-light) !important;
        border-top: none !important;
        background-color: var(--color-bg-primary) !important;
    }
    
    /* =================================
       Metric 组件
       ================================= */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: var(--font-size-h1) !important;
        font-weight: var(--font-weight-semibold) !important;
        color: var(--color-text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--color-text-muted) !important;
        font-size: var(--font-size-caption) !important;
        font-weight: var(--font-weight-medium) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: var(--font-size-micro) !important;
    }
    
    /* =================================
       自定义组件样式
       ================================= */
    
    /* Token 容器 */
    .token-container {
        background-color: var(--color-bg-tertiary);
        border: 1px solid var(--color-border-light);
        border-radius: var(--radius-md);
        padding: var(--spacing-lg);
        line-height: 2.2;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Token 块 */
    .token {
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        border-radius: var(--radius-sm);
        transition: transform var(--transition-fast);
        color: var(--color-text-primary) !important;
        font-weight: var(--font-weight-medium);
        font-size: var(--font-size-sm);
    }
    
    .token:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    
    /* 统计卡片 */
    .stat-card {
        background-color: var(--color-bg-tertiary);
        border: 1px solid var(--color-border-light);
        border-radius: var(--radius-md);
        padding: var(--spacing-lg);
        text-align: center;
    }
    
    .stat-value {
        font-size: var(--font-size-h1);
        font-weight: var(--font-weight-semibold);
        color: var(--color-accent);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stat-label {
        color: var(--color-text-muted);
        font-size: var(--font-size-caption);
        font-weight: var(--font-weight-medium);
        margin-top: var(--spacing-xs);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    /* 信息面板 */
    .info-panel {
        background-color: var(--color-bg-tertiary);
        border: 1px solid var(--color-border-light);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: var(--spacing-sm) 0;
        border-bottom: 1px solid var(--color-border-light);
    }
    
    .info-row:last-child {
        border-bottom: none;
    }
    
    .info-key {
        color: var(--color-text-muted);
        font-size: var(--font-size-sm);
    }
    
    .info-value {
        color: var(--color-text-primary);
        font-family: 'JetBrains Mono', monospace;
        font-size: var(--font-size-sm);
    }
    
    /* 提示框 */
    .tip-box {
        padding: var(--spacing-md) var(--spacing-lg);
        border-radius: var(--radius-md);
        margin: var(--spacing-lg) 0;
        font-size: var(--font-size-sm);
        background-color: var(--color-accent-light);
        border: 1px solid #93C5FD;
        color: var(--color-accent);
    }
    
    .warning-box {
        padding: var(--spacing-md) var(--spacing-lg);
        border-radius: var(--radius-md);
        margin: var(--spacing-lg) 0;
        font-size: var(--font-size-sm);
        background-color: var(--color-warning-light);
        border: 1px solid #FCD34D;
        color: var(--color-warning);
    }
    
    /* =================================
       侧边栏
       ================================= */
    [data-testid="stSidebar"] {
        background-color: var(--color-bg-secondary);
        border-right: 1px solid var(--color-border-light);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: var(--font-size-h3) !important;
        color: var(--color-text-primary) !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: var(--color-text-secondary);
        font-size: var(--font-size-sm);
    }
    
    [data-testid="stSidebar"] hr {
        border-color: var(--color-border-light);
        margin: var(--spacing-lg) 0;
    }
    
    /* =================================
       分隔线
       ================================= */
    hr {
        border-color: var(--color-border-light) !important;
        margin: var(--spacing-xl) 0 !important;
    }
    
    /* =================================
       代码块
       ================================= */
    code {
        background-color: var(--color-bg-tertiary) !important;
        color: var(--color-text-primary) !important;
        padding: 2px var(--spacing-sm);
        border-radius: var(--radius-sm);
        font-family: 'JetBrains Mono', monospace;
        font-size: var(--font-size-sm);
    }
    
    pre {
        background-color: var(--color-bg-tertiary) !important;
        border: 1px solid var(--color-border-light) !important;
        border-radius: var(--radius-md) !important;
    }
    
    /* =================================
       DataFrame / Table
       ================================= */
    .stDataFrame {
        border: 1px solid var(--color-border-light) !important;
        border-radius: var(--radius-md) !important;
    }
    
    /* =================================
       Alert / Info 框
       ================================= */
    .stAlert {
        border-radius: var(--radius-md) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        color: var(--color-accent) !important;
    }
</style>
"""

SIDEBAR_CSS = ""  # 已合并到 GLOBAL_CSS


def render_stat_card(value: str, label: str) -> str:
    """渲染统计卡片"""
    return f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def render_info_panel(items: dict) -> str:
    """渲染信息面板"""
    rows = []
    for key, value in items.items():
        rows.append(f"""
        <div class="info-row">
            <span class="info-key">{key}</span>
            <span class="info-value">{value}</span>
        </div>
        """)
    return f'<div class="info-panel">{"".join(rows)}</div>'
