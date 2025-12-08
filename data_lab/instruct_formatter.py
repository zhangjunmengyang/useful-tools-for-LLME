"""
格式化转换器 - SFT 数据格式转换
"""

import streamlit as st
import json
from data_lab.data_utils import CHAT_TEMPLATES, convert_to_format, validate_chat_format


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">格式化转换器</h1>', unsafe_allow_html=True)
    
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 输入数据")
        
        input_json = st.text_area(
            "原始 JSON",
            value='''{
    "instruction": "将以下句子翻译成英文",
    "input": "今天天气真好",
    "output": "The weather is really nice today."
}''',
            height=200
        )
        
        # 解析 JSON
        try:
            data = json.loads(input_json)
        except json.JSONDecodeError as e:
            st.error(f"JSON 格式错误: {e}")
            data = None
    
    with col_right:
        st.markdown("### 输出格式")
        
        target_format = st.selectbox(
            "目标格式",
            options=list(CHAT_TEMPLATES.keys()),
            format_func=lambda x: CHAT_TEMPLATES[x]['name']
        )
        
        system_prompt = st.text_input(
            "System Prompt (可选)",
            placeholder="自定义 system 提示词"
        )
        
        if data:
            converted = convert_to_format(data, target_format, system_prompt)
            
            st.text_area("转换结果", value=converted, height=200)
            
            # 格式验证
            validation = validate_chat_format(converted, target_format)
            if not validation['valid']:
                st.warning("格式问题:")
                for issue in validation['issues']:
                    st.caption(f"- {issue}")    


