"""
Chat Template - 对话模版调试器
"""

import json
import streamlit as st
import pandas as pd
from token_lab.tokenizer_utils import (
    load_tokenizer,
    apply_chat_template_safe,
    get_token_info,
    get_models_by_category,
    MODEL_CATEGORIES,
    TOKEN_COLORS
)


def render_template_output(template_str: str, tokenizer):
    """渲染模版输出"""
    if not template_str:
        return
    
    special_tokens = set()
    if hasattr(tokenizer, 'all_special_tokens'):
        special_tokens.update(tokenizer.all_special_tokens)
    if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
        special_tokens.update(tokenizer.additional_special_tokens)
    
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    
    result_html = ['<div style="font-family: \'JetBrains Mono\', monospace; white-space: pre-wrap; '
                   'background: #F3F4F6; padding: 16px; border-radius: 6px; '
                   'line-height: 1.8; font-size: 13px; border: 1px solid #E5E7EB; color: #111827;">']
    
    i = 0
    while i < len(template_str):
        found = False
        for special in sorted_special:
            if template_str[i:].startswith(special):
                escaped = special.replace('<', '&lt;').replace('>', '&gt;')
                result_html.append(
                    f'<span style="background: #FEE2E2; color: #DC2626; '
                    f'padding: 2px 4px; border-radius: 3px; border: 1px solid #FECACA; font-weight: 500;">'
                    f'{escaped}</span>'
                )
                i += len(special)
                found = True
                break
        
        if not found:
            char = template_str[i]
            if char == '<':
                result_html.append('&lt;')
            elif char == '>':
                result_html.append('&gt;')
            elif char == '\n':
                result_html.append('<br/>')
            elif char == ' ':
                result_html.append('&nbsp;')
            else:
                result_html.append(char)
            i += 1
    
    result_html.append('</div>')
    st.markdown(''.join(result_html), unsafe_allow_html=True)


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Chat Template</h1>', unsafe_allow_html=True)
    
    # 两级联动模型选择
    st.markdown("#### 选择模型")
    
    categories = list(MODEL_CATEGORIES.keys())
    category_display = [cat for cat in categories]
    
    col_provider, col_model = st.columns([1, 2])
    
    with col_provider:
        selected_category_display = st.selectbox(
            "模型厂商",
            options=category_display,
            index=0,
            key="chat_provider",
            help="选择模型提供商"
        )
        selected_category = categories[category_display.index(selected_category_display)]
    
    with col_model:
        models = get_models_by_category(selected_category)
        model_display = [name for name, _ in models]
        model_ids = [model_id for _, model_id in models]
        
        selected_model_display = st.selectbox(
            "选择模型",
            options=model_display,
            index=0,
            key="chat_model",
            help="选择该厂商下的具体模型"
        )
        
        if model_display:
            model_idx = model_display.index(selected_model_display)
            model_choice = model_ids[model_idx]
        else:
            model_choice = None
    
    if model_choice:
        st.caption(f"Tokenizer: `{model_choice}`")
    
    tokenizer = load_tokenizer(model_choice) if model_choice else None
    
    if not tokenizer:
        st.error("模型加载失败")
        return
    
    has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    
    if not has_template:
        st.warning(f"模型 {model_choice} 没有 Chat Template")
    
    st.markdown("---")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("#### 对话输入")
        
        default_json = json.dumps([{"role": "user", "content": ""}], indent=2, ensure_ascii=False)
        
        json_input = st.text_area("JSON", value=default_json, height=250, key="chat_json", 
                                   label_visibility="collapsed", placeholder="OpenAI 格式 JSON")
        
        add_gen_prompt = st.checkbox("添加 generation prompt", value=True, key="add_gen")
    
    with col_output:
        st.markdown("#### 渲染结果")
        
        if json_input:
            try:
                messages = json.loads(json_input)
                has_content = any(msg.get('content') for msg in messages if isinstance(msg, dict))
                
                if has_content:
                    if not isinstance(messages, list):
                        st.error("JSON 必须是数组")
                        return
                    
                    for i, msg in enumerate(messages):
                        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                            st.error(f"消息 {i} 格式错误")
                            return
                    
                    rendered, error = apply_chat_template_safe(tokenizer, messages, add_generation_prompt=add_gen_prompt)
                    
                    if error:
                        st.error(error)
                        st.markdown("#### 消息结构")
                        for msg in messages:
                            role_color = {"system": "#D97706", "user": "#2563EB", "assistant": "#059669"}.get(msg['role'], "#6B7280")
                            st.markdown(
                                f'<div style="border-left: 3px solid {role_color}; padding-left: 12px; margin: 8px 0; '
                                f'background: #F3F4F6; padding: 8px 12px; border-radius: 0 6px 6px 0;">'
                                f'<strong style="color: {role_color};">{msg["role"]}</strong><br/>'
                                f'<span style="color: #111827;">{msg["content"]}</span></div>',
                                unsafe_allow_html=True
                            )
                    else:
                        render_template_output(rendered, tokenizer)
                        
                        token_info = get_token_info(tokenizer, rendered)
                        
                        st.markdown("---")
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Tokens", len(token_info))
                        with c2:
                            st.metric("Special", sum(1 for t in token_info if t.get('is_special')))
                        with c3:
                            st.metric("Chars", len(rendered))
                        
                        with st.expander("Token 序列"):
                            html = ['<div style="line-height: 2.2;">']
                            for idx, info in enumerate(token_info[:200]):
                                is_special = info.get('is_special', False)
                                bg = '#FEE2E2' if is_special else TOKEN_COLORS[idx % len(TOKEN_COLORS)]
                                border = 'border: 1px solid #DC2626;' if is_special else ''
                                
                                display = info['token_str'].replace(' ', '\u2423').replace('\n', '\u21b5')
                                if not display.strip():
                                    display = repr(info['token_str'])[1:-1] or '[E]'
                                display = display.replace('<', '&lt;').replace('>', '&gt;')[:20]
                                
                                html.append(
                                    f'<span style="background:{bg}; color:#111827; '
                                    f'padding:2px 5px; margin:1px; border-radius:3px; display:inline-block; '
                                    f'font-family:monospace; font-size:13px; {border}">{display}</span>'
                                )
                            
                            if len(token_info) > 200:
                                html.append(f'<span style="color:#6B7280;">... +{len(token_info)-200}</span>')
                            
                            html.append('</div>')
                            st.markdown(''.join(html), unsafe_allow_html=True)
                        
                        with st.expander("原始字符串"):
                            st.code(rendered, language=None)
                        
                        with st.expander("Token IDs"):
                            st.code(str([t['token_id'] for t in token_info]), language="python")
                
            except json.JSONDecodeError:
                pass
    
    st.markdown("---")
    with st.expander("Chat Template 源码"):
        if has_template:
            st.code(tokenizer.chat_template, language="jinja2")
        else:
            st.info("无 Chat Template")

