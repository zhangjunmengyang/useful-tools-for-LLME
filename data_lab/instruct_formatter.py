"""
格式化转换器 - SFT 数据格式转换
"""

import streamlit as st
import json
from data_lab.data_utils import CHAT_TEMPLATES, convert_to_format, validate_chat_format


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">格式化转换器</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 一键转换 SFT 数据格式，支持 Alpaca、ShareGPT、ChatML、Llama-2 等主流格式。
    </div>
    """, unsafe_allow_html=True)
    
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
            st.success("✅ JSON 格式正确")
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
            if validation['valid']:
                st.success("✅ 格式验证通过")
            else:
                st.warning("⚠️ 格式问题:")
                for issue in validation['issues']:
                    st.caption(f"- {issue}")
    
    # 格式说明
    st.markdown("---")
    st.markdown("### 📋 格式说明")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Alpaca", "ShareGPT", "ChatML", "Llama-2"])
    
    with tab1:
        st.code("""### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""", language="text")
    
    with tab2:
        st.code("""{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}""", language="json")
    
    with tab3:
        st.code("""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>""", language="text")
    
    with tab4:
        st.code("""<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant} </s>""", language="text")
    
    st.markdown("""
    ### ⚠️ 常见问题
    
    1. **EOS Token 处理**: 确保每条数据以正确的 EOS token 结尾
    2. **标签闭合**: ChatML/Llama 格式需要严格的标签闭合
    3. **指令注入**: 避免用户输入包含特殊标签导致格式混乱
    """)

