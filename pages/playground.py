"""
分词编码 - 交互式分词、编解码与字节分析
"""

import streamlit as st
import pandas as pd
from utils.tokenizer_utils import (
    load_tokenizer, 
    get_token_info, 
    get_tokenizer_info,
    decode_token_ids,
    calculate_compression_stats,
    get_normalization_info,
    get_unicode_info,
    get_special_tokens_map,
    get_model_categories,
    get_models_by_category,
    MODEL_CATEGORIES,
    TOKEN_COLORS
)


def render_token_display(token_info_list: list, show_ids: bool = True) -> None:
    """渲染分词结果"""
    if not token_info_list:
        st.info("请输入文本")
        return
    
    html_parts = ['<div style="background-color: #F3F4F6; border: 1px solid #E5E7EB; border-radius: 6px; padding: 16px; line-height: 2.2; font-family: \'JetBrains Mono\', monospace;">']
    
    for idx, info in enumerate(token_info_list):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        token_str = info['token_str']
        token_id = info['token_id']
        raw_token = info.get('raw_token', token_str)
        byte_seq = info.get('byte_sequence', 'N/A')
        is_special = info.get('is_special', False)
        is_byte_fallback = info.get('is_byte_fallback', False)
        is_byte_token = info.get('is_byte_token', False)
        
        display_str = token_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_str = display_str.replace(' ', '\u2423')
        if '\n' in display_str:
            display_str = display_str.replace('\n', '\u21b5')
        if '\t' in display_str:
            display_str = display_str.replace('\t', '\u2192')
        
        if not display_str.strip():
            display_str = repr(token_str)[1:-1] if token_str else '[EMPTY]'
        
        border_style = ""
        label = ""
        if is_special:
            border_style = "border: 2px solid #DC2626;"
            label = '<small style="color:#DC2626;font-size:10px;margin-left:2px;">[S]</small>'
        elif is_byte_fallback:
            border_style = "border: 2px dashed #D97706;"
            label = '<small style="color:#D97706;font-size:10px;margin-left:2px;">[B]</small>'
        elif is_byte_token:
            border_style = "border: 1px solid #9CA3AF;"
        
        tooltip_text = f"Token: {raw_token}  |  ID: {token_id}  |  Bytes: {byte_seq}"
        id_html = f'<sub style="font-size: 11px; color: #6B7280; margin-left: 2px;">{token_id}</sub>' if show_ids else ''
        
        html_parts.append(
            f'<span title="{tooltip_text}" '
            f'style="background-color: {color}; color: #111827; padding: 4px 8px; margin: 2px; '
            f'border-radius: 4px; display: inline-block; font-size: 13px; cursor: default; {border_style}">'
            f'{display_str}{id_html}{label}</span>'
        )
    
    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)


def render_unicode_table(text: str):
    """渲染 Unicode 表"""
    if not text:
        return
    
    data = []
    for char in text:
        info = get_unicode_info(char)
        data.append({
            "字符": char,
            "名称": info.get('name', 'N/A'),
            "分类": info.get('category', 'N/A'),
            "码点": info.get('codepoint', 'N/A'),
            "十进制": info.get('decimal', 'N/A'),
            "UTF-8": info.get('utf8_bytes', 'N/A')
        })
    
    st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">分词编码</h1>', unsafe_allow_html=True)
    
    # 两级联动模型选择
    st.markdown("#### 选择模型")
    
    # 获取厂商列表
    categories = list(MODEL_CATEGORIES.keys())
    category_display = [cat for cat in categories]
    
    col_provider, col_model = st.columns([1, 2])
    
    with col_provider:
        # 厂商选择
        selected_category_display = st.selectbox(
            "模型厂商",
            options=category_display,
            index=0,
            key="model_provider",
            help="选择模型提供商"
        )
        # 提取实际的分类名称（去掉图标）
        selected_category = categories[category_display.index(selected_category_display)]
    
    with col_model:
        # 获取该厂商下的模型
        models = get_models_by_category(selected_category)
        model_display = [name for name, _ in models]
        model_ids = [model_id for _, model_id in models]
        
        # 模型选择
        selected_model_display = st.selectbox(
            "选择模型",
            options=model_display,
            index=0,
            key="model_name",
            help="选择该厂商下的具体模型"
        )
        
        # 获取实际的 model_id
        if model_display:
            model_idx = model_display.index(selected_model_display)
            model_name = model_ids[model_idx]
        else:
            model_name = None
    
    # 显示选中的模型 ID
    if model_name:
        st.caption(f"📦 Tokenizer: `{model_name}`")
    
    tokenizer = load_tokenizer(model_name) if model_name else None
    
    if not tokenizer:
        st.warning("请选择模型")
        return
    
    # 模型信息
    with st.expander("模型信息", expanded=False):
        info = get_tokenizer_info(tokenizer)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("词表大小", f"{info['vocab_size']:,}")
        with col_b:
            st.metric("最大长度", str(info['model_max_length']))
        with col_c:
            st.metric("算法类型", info['algorithm'].split('(')[0].strip())
        
        special_tokens = []
        for name in ['bos_token', 'eos_token', 'pad_token', 'unk_token']:
            token = info.get(name)
            if token:
                special_tokens.append(f"**{name}**: `{token}`")
        if special_tokens:
            st.markdown(" | ".join(special_tokens))
    
    st.markdown("---")
    
    # 主要功能 Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["编码", "解码", "字节分析", "Unicode", "特殊 Token"])
    
    # ========== 编码 Tab ==========
    with tab1:
        input_text = st.text_area(
            "输入文本",
            value="",
            height=120,
            placeholder="输入文本，点击外部或按 Cmd+Enter 查看结果",
            key="encode_input"
        )
        
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            show_ids = st.checkbox("显示 Token ID", value=True, key="show_ids")
        with opt_col2:
            add_special = st.checkbox("添加特殊 Token", value=False, key="add_special")
        
        if input_text:
            if add_special:
                encoding = tokenizer(input_text, add_special_tokens=True)
                token_ids = encoding["input_ids"]
                token_info_list = []
                special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
                for idx, tid in enumerate(token_ids):
                    raw_token = tokenizer.convert_ids_to_tokens([tid])[0] if hasattr(tokenizer, 'convert_ids_to_tokens') else ""
                    token_str = tokenizer.decode([tid])
                    
                    is_byte_token = '\ufffd' in token_str
                    display_token_str = raw_token if (is_byte_token and raw_token) else token_str
                    
                    try:
                        byte_seq = ' '.join([f'0x{b:02X}' for b in (raw_token or token_str).encode('utf-8')])
                    except:
                        byte_seq = "N/A"
                    token_info_list.append({
                        "token_str": display_token_str,
                        "decoded_str": token_str,
                        "raw_token": raw_token,
                        "token_id": tid,
                        "byte_sequence": byte_seq,
                        "is_special": tid in special_ids,
                        "is_byte_fallback": raw_token.startswith('<0x') and raw_token.endswith('>'),
                        "is_byte_token": is_byte_token,
                        "index": idx
                    })
            else:
                token_info_list = get_token_info(tokenizer, input_text)
            
            stats = calculate_compression_stats(input_text, len(token_info_list))
            
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Token 数", stats['token_count'])
            with stat_cols[1]:
                st.metric("字符数", stats['char_count'])
            with stat_cols[2]:
                st.metric("字符/Token", stats['chars_per_token'])
            with stat_cols[3]:
                st.metric("字节/Token", stats['bytes_per_token'])
            
            st.markdown("#### 分词结果")
            render_token_display(token_info_list, show_ids)
            
            with st.expander("Token ID 序列"):
                ids = [info['token_id'] for info in token_info_list]
                st.code(str(ids), language="python")
            
            with st.expander("详细信息"):
                df_data = []
                for info in token_info_list:
                    df_data.append({
                        "Index": info['index'],
                        "Token": info['raw_token'],
                        "Decoded": repr(info['token_str']),
                        "ID": info['token_id'],
                        "Bytes": info['byte_sequence'],
                        "Special": "Yes" if info.get('is_special') else "",
                        "Fallback": "Yes" if info.get('is_byte_fallback') else ""
                    })
                st.dataframe(pd.DataFrame(df_data), width="stretch", hide_index=True)
    
    # ========== 解码 Tab ==========
    with tab2:
        id_input = st.text_input(
            "Token IDs",
            placeholder="例如: 128000, 50256 或 [128000, 50256]",
            key="decode_input"
        )
        
        if id_input:
            try:
                cleaned = id_input.strip().strip('[]')
                token_ids = [int(x.strip()) for x in cleaned.split(',') if x.strip()]
                
                if token_ids:
                    decoded_text, individual_tokens = decode_token_ids(tokenizer, token_ids)
                    
                    st.markdown("#### 解码结果")
                    st.code(decoded_text, language=None)
                    
                    st.markdown("#### Token 详情")
                    for tok in individual_tokens:
                        st.markdown(
                            f"**ID {tok['token_id']}** → `{tok['raw_token']}` → \"{tok['token_str']}\""
                        )
            except ValueError as e:
                st.error(f"格式错误: {str(e)}")
    
    # ========== 字节分析 Tab ==========
    with tab3:
        byte_input = st.text_area("输入文本", value="", height=80, 
                                   placeholder="输入 Emoji、生僻字等特殊字符查看字节级分词", key="byte_input")
        
        if byte_input:
            token_info = get_token_info(tokenizer, byte_input)
            
            total = len(token_info)
            fallback_count = sum(1 for t in token_info if t.get('is_byte_fallback', False))
            byte_token_count = sum(1 for t in token_info if t.get('is_byte_token', False))
            special_count = sum(1 for t in token_info if t.get('is_special', False))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总 Token", total)
            with col2:
                st.metric("Byte Fallback", fallback_count, 
                         delta="需要回退" if fallback_count > 0 else None, delta_color="off")
            with col3:
                st.metric("字节级 BPE", byte_token_count)
            with col4:
                st.metric("特殊 Token", special_count)
            
            st.markdown("#### 分词结果")
            render_token_display(token_info, show_ids=True)
            
            if fallback_count > 0:
                st.markdown("#### Byte Fallback 详情")
                fallback_data = [{"Index": t['index'], "Token": t['raw_token'], "ID": t['token_id'], 
                                 "Byte": t.get('byte_sequence', 'N/A')} 
                                for t in token_info if t.get('is_byte_fallback')]
                if fallback_data:
                    st.dataframe(pd.DataFrame(fallback_data), width="stretch", hide_index=True)
    
    # ========== Unicode Tab ==========
    with tab4:
        norm_input = st.text_input("输入文本", value="", placeholder="输入要分析的文本", key="norm_input")
        
        if norm_input:
            norm_info = get_normalization_info(norm_input)
            
            st.markdown("#### 规范化对比")
            norm_data = [
                {"形式": "原始", "长度": norm_info['original_len'], "文本": norm_info['original'], "相同": "-"},
                {"形式": "NFC", "长度": norm_info['NFC_len'], "文本": norm_info['NFC'], "相同": "Yes" if norm_info['nfc_equal'] else "No"},
                {"形式": "NFD", "长度": norm_info['NFD_len'], "文本": norm_info['NFD'], "相同": "Yes" if norm_info['nfd_equal'] else "No"},
                {"形式": "NFKC", "长度": norm_info['NFKC_len'], "文本": norm_info['NFKC'], "相同": "-"},
                {"形式": "NFKD", "长度": norm_info['NFKD_len'], "文本": norm_info['NFKD'], "相同": "-"},
            ]
            st.dataframe(pd.DataFrame(norm_data), width="stretch", hide_index=True)
            
            st.markdown("#### Unicode 详情")
            render_unicode_table(norm_input[:50])
    
    # ========== 特殊 Token Tab ==========
    with tab5:
        special_map = get_special_tokens_map(tokenizer)
        
        if special_map:
            st.markdown("#### 标准特殊 Token")
            standard = [{"名称": n, "Token": special_map[n]['token'], "ID": special_map[n]['id']} 
                       for n in ['bos_token', 'eos_token', 'pad_token', 'unk_token', 'cls_token', 'sep_token', 'mask_token'] 
                       if n in special_map]
            if standard:
                st.dataframe(pd.DataFrame(standard), width="stretch", hide_index=True)
            else:
                st.info("无标准特殊 Token")
            
            if 'additional_special_tokens' in special_map and special_map['additional_special_tokens']:
                st.markdown("#### 额外特殊 Token")
                additional = special_map['additional_special_tokens']
                st.dataframe(pd.DataFrame([{"Token": t['token'], "ID": t['id']} for t in additional[:20]]), 
                            width="stretch", hide_index=True)
                if len(additional) > 20:
                    st.caption(f"共 {len(additional)} 个")
        else:
            st.info("无法获取特殊 Token")
