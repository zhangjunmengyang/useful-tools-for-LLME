"""
分词编码 - Gradio 版本
交互式分词、编解码与字节分析
"""

import gradio as gr
import pandas as pd
from token_lab.tokenizer_utils import (
    load_tokenizer,
    get_token_info,
    get_tokenizer_info,
    decode_token_ids,
    calculate_compression_stats,
    get_normalization_info,
    get_unicode_info,
    get_special_tokens_map,
    get_models_by_category,
    MODEL_CATEGORIES,
    TOKEN_COLORS
)


def get_category_choices():
    """获取模型厂商列表"""
    return list(MODEL_CATEGORIES.keys())


def get_model_choices(category):
    """根据厂商获取模型列表"""
    if not category:
        return []
    models = get_models_by_category(category)
    return [name for name, _ in models]


def get_model_id(category, model_name):
    """获取模型 ID"""
    if not category or not model_name:
        return None
    models = get_models_by_category(category)
    for name, model_id in models:
        if name == model_name:
            return model_id
    return None


def render_tokens_html(token_info_list, show_ids=True):
    """渲染分词结果为 HTML"""
    if not token_info_list:
        return "<div style='color: #6B7280; padding: 20px;'>请输入文本</div>"
    
    html_parts = [
        '<div style="background-color: #F3F4F6; border: 1px solid #E5E7EB; '
        'border-radius: 8px; padding: 16px; line-height: 2.4; '
        'font-family: \'JetBrains Mono\', monospace;">'
    ]
    
    for idx, info in enumerate(token_info_list):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        token_str = info['token_str']
        token_id = info['token_id']
        raw_token = info.get('raw_token', token_str)
        is_special = info.get('is_special', False)
        is_byte_fallback = info.get('is_byte_fallback', False)
        
        # 转义和替换特殊字符
        display_str = token_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_str = display_str.replace(' ', '␣')
        display_str = display_str.replace('\n', '↵')
        display_str = display_str.replace('\t', '→')
        
        if not display_str.strip():
            display_str = repr(token_str)[1:-1] if token_str else '[EMPTY]'
        
        # 样式
        border_style = ""
        label = ""
        if is_special:
            border_style = "border: 2px solid #DC2626;"
            label = '<span style="color:#DC2626;font-size:10px;margin-left:2px;">[S]</span>'
        elif is_byte_fallback:
            border_style = "border: 2px dashed #D97706;"
            label = '<span style="color:#D97706;font-size:10px;margin-left:2px;">[B]</span>'
        
        id_html = f'<sub style="font-size: 10px; color: #6B7280; margin-left: 3px;">{token_id}</sub>' if show_ids else ''
        
        html_parts.append(
            f'<span title="Token: {raw_token} | ID: {token_id}" '
            f'style="background-color: {color}; color: #111827; padding: 4px 8px; margin: 2px; '
            f'border-radius: 4px; display: inline-block; font-size: 13px; cursor: default; {border_style}">'
            f'{display_str}{id_html}{label}</span>'
        )
    
    html_parts.append('</div>')
    return ''.join(html_parts)


def encode_text(category, model_name, text, show_ids, add_special):
    """编码文本"""
    if not category or not model_name:
        return (
            "<div style='color: #DC2626;'>请选择模型</div>",
            "", "", "", "",
            pd.DataFrame()
        )
    
    model_id = get_model_id(category, model_name)
    if not model_id:
        return (
            "<div style='color: #DC2626;'>模型不存在</div>",
            "", "", "", "",
            pd.DataFrame()
        )
    
    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return (
            "<div style='color: #DC2626;'>模型加载失败</div>",
            "", "", "", "",
            pd.DataFrame()
        )
    
    if not text:
        return (
            render_tokens_html([], show_ids),
            "0", "0", "-", "-",
            pd.DataFrame()
        )
    
    # 获取 token 信息
    if add_special:
        encoding = tokenizer(text, add_special_tokens=True)
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
                "is_byte_fallback": raw_token.startswith('<0x') and raw_token.endswith('>') if raw_token else False,
                "is_byte_token": is_byte_token,
                "index": idx
            })
    else:
        token_info_list = get_token_info(tokenizer, text)
    
    # 计算统计
    stats = calculate_compression_stats(text, len(token_info_list))
    
    # Token IDs
    ids = [info['token_id'] for info in token_info_list]
    
    # 详细信息表格
    df_data = []
    for info in token_info_list:
        df_data.append({
            "Index": info['index'],
            "Token": info['raw_token'],
            "Decoded": repr(info['token_str']),
            "ID": info['token_id'],
            "Bytes": info.get('byte_sequence', 'N/A'),
            "Special": "✓" if info.get('is_special') else "",
            "Fallback": "✓" if info.get('is_byte_fallback') else ""
        })
    
    return (
        render_tokens_html(token_info_list, show_ids),
        str(stats['token_count']),
        str(stats['char_count']),
        str(stats['chars_per_token']),
        str(stats['bytes_per_token']),
        pd.DataFrame(df_data)
    )


def decode_ids(category, model_name, id_input):
    """解码 Token IDs"""
    if not category or not model_name:
        return "请选择模型", ""
    
    model_id = get_model_id(category, model_name)
    if not model_id:
        return "模型不存在", ""
    
    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return "模型加载失败", ""
    
    if not id_input:
        return "", ""
    
    try:
        cleaned = id_input.strip().strip('[]')
        token_ids = [int(x.strip()) for x in cleaned.split(',') if x.strip()]
        
        if not token_ids:
            return "请输入有效的 Token IDs", ""
        
        decoded_text, individual_tokens = decode_token_ids(tokenizer, token_ids)
        
        details = []
        for tok in individual_tokens:
            details.append(f"**ID {tok['token_id']}** → `{tok['raw_token']}` → \"{tok['token_str']}\"")
        
        return decoded_text, "\n\n".join(details)
    
    except ValueError as e:
        return f"格式错误: {str(e)}", ""


def analyze_bytes(category, model_name, text):
    """字节分析"""
    if not category or not model_name:
        return "<div>请选择模型</div>", "0", "0", "0", "0", pd.DataFrame()
    
    model_id = get_model_id(category, model_name)
    if not model_id:
        return "<div>模型不存在</div>", "0", "0", "0", "0", pd.DataFrame()
    
    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return "<div>模型加载失败</div>", "0", "0", "0", "0", pd.DataFrame()
    
    if not text:
        return render_tokens_html([], True), "0", "0", "0", "0", pd.DataFrame()
    
    token_info = get_token_info(tokenizer, text)
    
    total = len(token_info)
    fallback_count = sum(1 for t in token_info if t.get('is_byte_fallback', False))
    byte_token_count = sum(1 for t in token_info if t.get('is_byte_token', False))
    special_count = sum(1 for t in token_info if t.get('is_special', False))
    
    # Fallback 详情
    fallback_data = []
    for t in token_info:
        if t.get('is_byte_fallback'):
            fallback_data.append({
                "Index": t['index'],
                "Token": t['raw_token'],
                "ID": t['token_id'],
                "Byte": t.get('byte_sequence', 'N/A')
            })
    
    return (
        render_tokens_html(token_info, True),
        str(total),
        str(fallback_count),
        str(byte_token_count),
        str(special_count),
        pd.DataFrame(fallback_data) if fallback_data else pd.DataFrame()
    )


def analyze_unicode(text):
    """Unicode 分析"""
    if not text:
        return pd.DataFrame(), pd.DataFrame()
    
    # 规范化对比
    norm_info = get_normalization_info(text)
    norm_data = [
        {"形式": "原始", "长度": norm_info['original_len'], "文本": norm_info['original'], "相同": "-"},
        {"形式": "NFC", "长度": norm_info['NFC_len'], "文本": norm_info['NFC'], "相同": "✓" if norm_info['nfc_equal'] else "✗"},
        {"形式": "NFD", "长度": norm_info['NFD_len'], "文本": norm_info['NFD'], "相同": "✓" if norm_info['nfd_equal'] else "✗"},
        {"形式": "NFKC", "长度": norm_info['NFKC_len'], "文本": norm_info['NFKC'], "相同": "-"},
        {"形式": "NFKD", "长度": norm_info['NFKD_len'], "文本": norm_info['NFKD'], "相同": "-"},
    ]
    
    # Unicode 详情
    unicode_data = []
    for char in text[:50]:
        info = get_unicode_info(char)
        unicode_data.append({
            "字符": char,
            "名称": info.get('name', 'N/A'),
            "分类": info.get('category', 'N/A'),
            "码点": info.get('codepoint', 'N/A'),
            "十进制": info.get('decimal', 'N/A'),
            "UTF-8": info.get('utf8_bytes', 'N/A')
        })
    
    return pd.DataFrame(norm_data), pd.DataFrame(unicode_data)


def get_special_tokens(category, model_name):
    """获取特殊 Token"""
    if not category or not model_name:
        return pd.DataFrame(), pd.DataFrame()
    
    model_id = get_model_id(category, model_name)
    if not model_id:
        return pd.DataFrame(), pd.DataFrame()
    
    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return pd.DataFrame(), pd.DataFrame()
    
    special_map = get_special_tokens_map(tokenizer)
    
    if not special_map:
        return pd.DataFrame(), pd.DataFrame()
    
    # 标准特殊 Token
    standard = []
    for name in ['bos_token', 'eos_token', 'pad_token', 'unk_token', 'cls_token', 'sep_token', 'mask_token']:
        if name in special_map:
            standard.append({
                "名称": name,
                "Token": special_map[name]['token'],
                "ID": special_map[name]['id']
            })
    
    # 额外特殊 Token
    additional = []
    if 'additional_special_tokens' in special_map and special_map['additional_special_tokens']:
        for t in special_map['additional_special_tokens'][:20]:
            additional.append({"Token": t['token'], "ID": t['id']})
    
    return pd.DataFrame(standard), pd.DataFrame(additional)


def get_model_info(category, model_name):
    """获取模型信息"""
    if not category or not model_name:
        return "请选择模型"
    
    model_id = get_model_id(category, model_name)
    if not model_id:
        return "模型不存在"
    
    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return "模型加载失败"
    
    info = get_tokenizer_info(tokenizer)
    
    return f"""
**Tokenizer**: `{model_id}`

| 属性 | 值 |
|------|-----|
| 词表大小 | {info['vocab_size']:,} |
| 最大长度 | {info['model_max_length']} |
| 算法类型 | {info['algorithm'].split('(')[0].strip()} |
| BOS Token | `{info.get('bos_token', 'N/A')}` |
| EOS Token | `{info.get('eos_token', 'N/A')}` |
| PAD Token | `{info.get('pad_token', 'N/A')}` |
| UNK Token | `{info.get('unk_token', 'N/A')}` |
"""


def render():
    """渲染页面"""
    
    gr.Markdown("## 分词编码")
    
    # 模型选择
    initial_category = get_category_choices()[0] if get_category_choices() else None
    initial_models = get_model_choices(initial_category) if initial_category else []
    
    with gr.Row():
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=get_category_choices(),
                label="模型厂商",
                value=initial_category,
                interactive=True
            )
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                label="选择模型",
                value=initial_models[0] if initial_models else None,
                interactive=True
            )
    
    # 模型 ID 显示
    model_id_text = gr.Markdown("", elem_id="model-id-display")
    
    # 模型信息折叠
    with gr.Accordion("模型信息", open=False):
        model_info_md = gr.Markdown("")
    
    gr.Markdown("---")
    
    # 功能 Tabs
    with gr.Tabs() as tabs:
        
        # ========== 编码 Tab ==========
        with gr.Tab("编码"):
            encode_input = gr.Textbox(
                label="输入文本",
                placeholder="输入文本进行分词编码...",
                lines=4
            )
            
            with gr.Row():
                show_ids_cb = gr.Checkbox(label="显示 Token ID", value=True)
                add_special_cb = gr.Checkbox(label="添加特殊 Token", value=False)
            
            with gr.Row():
                with gr.Column(scale=1):
                    stat_tokens = gr.Textbox(label="Token 数", interactive=False)
                with gr.Column(scale=1):
                    stat_chars = gr.Textbox(label="字符数", interactive=False)
                with gr.Column(scale=1):
                    stat_chars_per_token = gr.Textbox(label="字符/Token", interactive=False)
                with gr.Column(scale=1):
                    stat_bytes_per_token = gr.Textbox(label="字节/Token", interactive=False)
            
            gr.Markdown("### 分词结果")
            tokens_html = gr.HTML("")
            
            with gr.Accordion("详细信息", open=False):
                detail_df = gr.Dataframe(
                    headers=["Index", "Token", "Decoded", "ID", "Bytes", "Special", "Fallback"],
                    interactive=False
                )
        
        # ========== 解码 Tab ==========
        with gr.Tab("解码"):
            decode_input = gr.Textbox(
                label="Token IDs",
                placeholder="例如: 128000, 50256 或 [128000, 50256]",
                lines=1
            )
            
            decode_btn = gr.Button("解码", variant="primary")
            
            gr.Markdown("### 解码结果")
            decode_result = gr.Textbox(label="解码文本", interactive=False, lines=3)
            decode_details = gr.Markdown("")
        
        # ========== 字节分析 Tab ==========
        with gr.Tab("字节分析"):
            byte_input = gr.Textbox(
                label="输入文本",
                placeholder="输入 Emoji、生僻字等特殊字符查看字节级分词",
                lines=2
            )
            
            byte_analyze_btn = gr.Button("分析", variant="primary")
            
            with gr.Row():
                byte_total = gr.Textbox(label="总 Token", interactive=False)
                byte_fallback = gr.Textbox(label="Byte Fallback", interactive=False)
                byte_bpe = gr.Textbox(label="字节级 BPE", interactive=False)
                byte_special = gr.Textbox(label="特殊 Token", interactive=False)
            
            gr.Markdown("### 分词结果")
            byte_tokens_html = gr.HTML("")
            
            with gr.Accordion("Byte Fallback 详情", open=False):
                byte_fallback_df = gr.Dataframe(interactive=False)
        
        # ========== Unicode Tab ==========
        with gr.Tab("Unicode"):
            unicode_input = gr.Textbox(
                label="输入文本",
                placeholder="输入要分析的文本",
                lines=1
            )
            
            unicode_btn = gr.Button("分析", variant="primary")
            
            gr.Markdown("### 规范化对比")
            norm_df = gr.Dataframe(interactive=False)
            
            gr.Markdown("### Unicode 详情")
            unicode_df = gr.Dataframe(interactive=False)
        
        # ========== 特殊 Token Tab ==========
        with gr.Tab("特殊 Token"):
            special_btn = gr.Button("获取特殊 Token", variant="primary")
            
            gr.Markdown("### 标准特殊 Token")
            standard_df = gr.Dataframe(interactive=False)
            
            gr.Markdown("### 额外特殊 Token")
            additional_df = gr.Dataframe(interactive=False)
    
    # ==================== 事件绑定 ====================
    
    # 厂商变化 -> 更新模型列表
    def update_models(category):
        models = get_model_choices(category)
        return gr.update(choices=models, value=models[0] if models else None)
    
    category_dropdown.change(
        fn=update_models,
        inputs=[category_dropdown],
        outputs=[model_dropdown]
    )
    
    # 模型变化 -> 更新模型 ID 和信息
    def on_model_change(category, model_name):
        model_id = get_model_id(category, model_name)
        id_text = f"**Tokenizer**: `{model_id}`" if model_id else ""
        info = get_model_info(category, model_name)
        return id_text, info
    
    model_dropdown.change(
        fn=on_model_change,
        inputs=[category_dropdown, model_dropdown],
        outputs=[model_id_text, model_info_md]
    )
    
    # 编码功能
    encode_inputs = [category_dropdown, model_dropdown, encode_input, show_ids_cb, add_special_cb]
    encode_outputs = [tokens_html, stat_tokens, stat_chars, stat_chars_per_token, stat_bytes_per_token, detail_df]
    
    encode_input.change(fn=encode_text, inputs=encode_inputs, outputs=encode_outputs)
    show_ids_cb.change(fn=encode_text, inputs=encode_inputs, outputs=encode_outputs)
    add_special_cb.change(fn=encode_text, inputs=encode_inputs, outputs=encode_outputs)
    
    # 解码功能
    decode_btn.click(
        fn=decode_ids,
        inputs=[category_dropdown, model_dropdown, decode_input],
        outputs=[decode_result, decode_details]
    )
    
    # 字节分析
    byte_analyze_btn.click(
        fn=analyze_bytes,
        inputs=[category_dropdown, model_dropdown, byte_input],
        outputs=[byte_tokens_html, byte_total, byte_fallback, byte_bpe, byte_special, byte_fallback_df]
    )
    
    # Unicode 分析
    unicode_btn.click(
        fn=analyze_unicode,
        inputs=[unicode_input],
        outputs=[norm_df, unicode_df]
    )
    
    # 特殊 Token
    special_btn.click(
        fn=get_special_tokens,
        inputs=[category_dropdown, model_dropdown],
        outputs=[standard_df, additional_df]
    )
