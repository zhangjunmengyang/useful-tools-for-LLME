"""
Tokenizer Playground - Gradio Version
Interactive tokenization, encoding/decoding and byte analysis
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
        return "<div style='color: #6B7280; padding: 20px;'>Enter text to tokenize</div>"

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
            "<div style='color: #DC2626;'>Please select a model</div>",
            "", "", "", "",
            pd.DataFrame()
        )

    model_id = get_model_id(category, model_name)
    if not model_id:
        return (
            "<div style='color: #DC2626;'>Model not found</div>",
            "", "", "", "",
            pd.DataFrame()
        )

    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return (
            "<div style='color: #DC2626;'>Failed to load model</div>",
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

    stats = calculate_compression_stats(text, len(token_info_list))

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
        return "Please select a model", ""

    model_id = get_model_id(category, model_name)
    if not model_id:
        return "Model not found", ""

    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return "Failed to load model", ""

    if not id_input:
        return "", ""

    try:
        cleaned = id_input.strip().strip('[]')
        token_ids = [int(x.strip()) for x in cleaned.split(',') if x.strip()]

        if not token_ids:
            return "Please enter valid Token IDs", ""

        decoded_text, individual_tokens = decode_token_ids(tokenizer, token_ids)

        details = []
        for tok in individual_tokens:
            details.append(f"**ID {tok['token_id']}** → `{tok['raw_token']}` → \"{tok['token_str']}\"")

        return decoded_text, "\n\n".join(details)

    except ValueError as e:
        return f"Format error: {str(e)}", ""


def analyze_bytes(category, model_name, text):
    """字节分析"""
    if not category or not model_name:
        return "<div>Please select a model</div>", "0", "0", "0", "0", pd.DataFrame()

    model_id = get_model_id(category, model_name)
    if not model_id:
        return "<div>Model not found</div>", "0", "0", "0", "0", pd.DataFrame()

    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return "<div>Failed to load model</div>", "0", "0", "0", "0", pd.DataFrame()

    if not text:
        return render_tokens_html([], True), "0", "0", "0", "0", pd.DataFrame()

    token_info = get_token_info(tokenizer, text)

    total = len(token_info)
    fallback_count = sum(1 for t in token_info if t.get('is_byte_fallback', False))
    byte_token_count = sum(1 for t in token_info if t.get('is_byte_token', False))
    special_count = sum(1 for t in token_info if t.get('is_special', False))

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

    norm_info = get_normalization_info(text)
    norm_data = [
        {"Form": "Original", "Length": norm_info['original_len'], "Text": norm_info['original'], "Same": "-"},
        {"Form": "NFC", "Length": norm_info['NFC_len'], "Text": norm_info['NFC'], "Same": "✓" if norm_info['nfc_equal'] else "✗"},
        {"Form": "NFD", "Length": norm_info['NFD_len'], "Text": norm_info['NFD'], "Same": "✓" if norm_info['nfd_equal'] else "✗"},
        {"Form": "NFKC", "Length": norm_info['NFKC_len'], "Text": norm_info['NFKC'], "Same": "-"},
        {"Form": "NFKD", "Length": norm_info['NFKD_len'], "Text": norm_info['NFKD'], "Same": "-"},
    ]

    unicode_data = []
    for char in text[:50]:
        info = get_unicode_info(char)
        unicode_data.append({
            "Char": char,
            "Name": info.get('name', 'N/A'),
            "Category": info.get('category', 'N/A'),
            "Codepoint": info.get('codepoint', 'N/A'),
            "Decimal": info.get('decimal', 'N/A'),
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

    standard = []
    for name in ['bos_token', 'eos_token', 'pad_token', 'unk_token', 'cls_token', 'sep_token', 'mask_token']:
        if name in special_map:
            standard.append({
                "Name": name,
                "Token": special_map[name]['token'],
                "ID": special_map[name]['id']
            })

    additional = []
    if 'additional_special_tokens' in special_map and special_map['additional_special_tokens']:
        for t in special_map['additional_special_tokens'][:20]:
            additional.append({"Token": t['token'], "ID": t['id']})

    return pd.DataFrame(standard), pd.DataFrame(additional)


def get_model_info(category, model_name):
    """获取模型信息"""
    if not category or not model_name:
        return "Please select a model"

    model_id = get_model_id(category, model_name)
    if not model_id:
        return "Model not found"

    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return "Failed to load model"

    info = get_tokenizer_info(tokenizer)

    return f"""
**Tokenizer**: `{model_id}`

| Attribute | Value |
|-----------|-------|
| Vocab Size | {info['vocab_size']:,} |
| Max Length | {info['model_max_length']} |
| Algorithm | {info['algorithm'].split('(')[0].strip()} |
| BOS Token | `{info.get('bos_token', 'N/A')}` |
| EOS Token | `{info.get('eos_token', 'N/A')}` |
| PAD Token | `{info.get('pad_token', 'N/A')}` |
| UNK Token | `{info.get('unk_token', 'N/A')}` |
"""


def render():
    """渲染页面"""

    initial_category = get_category_choices()[0] if get_category_choices() else None
    initial_models = get_model_choices(initial_category) if initial_category else []

    with gr.Row():
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=get_category_choices(),
                label="Vendor",
                value=initial_category,
                interactive=True
            )
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                label="Model",
                value=initial_models[0] if initial_models else None,
                interactive=True
            )

    model_id_text = gr.Markdown("", elem_id="model-id-display")

    with gr.Accordion("Model Info", open=False):
        model_info_md = gr.Markdown("")

    gr.Markdown("---")

    with gr.Tabs() as tabs:

        # Encoding Tab
        with gr.Tab("Encode"):
            encode_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to tokenize...",
                lines=4
            )

            with gr.Row():
                show_ids_cb = gr.Checkbox(label="Show Token IDs", value=True)
                add_special_cb = gr.Checkbox(label="Add Special Tokens", value=False)

            with gr.Row():
                with gr.Column(scale=1):
                    stat_tokens = gr.Textbox(label="Tokens", interactive=False)
                with gr.Column(scale=1):
                    stat_chars = gr.Textbox(label="Characters", interactive=False)
                with gr.Column(scale=1):
                    stat_chars_per_token = gr.Textbox(label="Chars/Token", interactive=False)
                with gr.Column(scale=1):
                    stat_bytes_per_token = gr.Textbox(label="Bytes/Token", interactive=False)

            gr.Markdown("### Tokenization Result")
            tokens_html = gr.HTML("")

            with gr.Accordion("Details", open=False):
                detail_df = gr.Dataframe(
                    headers=["Index", "Token", "Decoded", "ID", "Bytes", "Special", "Fallback"],
                    interactive=False
                )

        # Decoding Tab
        with gr.Tab("Decode"):
            decode_input = gr.Textbox(
                label="Token IDs",
                placeholder="e.g., 128000, 50256 or [128000, 50256]",
                lines=1
            )

            gr.Markdown("### Decoded Result")
            decode_result = gr.Textbox(label="Decoded Text", interactive=False, lines=3)
            decode_details = gr.Markdown("")

        # Byte Analysis Tab
        with gr.Tab("Byte Analysis"):
            byte_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text with emojis or rare characters...",
                lines=2
            )

            with gr.Row():
                byte_total = gr.Textbox(label="Total Tokens", interactive=False)
                byte_fallback = gr.Textbox(label="Byte Fallback", interactive=False)
                byte_bpe = gr.Textbox(label="Byte-Level BPE", interactive=False)
                byte_special = gr.Textbox(label="Special Tokens", interactive=False)

            gr.Markdown("### Tokenization Result")
            byte_tokens_html = gr.HTML("")

            with gr.Accordion("Byte Fallback Details", open=False):
                byte_fallback_df = gr.Dataframe(interactive=False)

        # Unicode Tab
        with gr.Tab("Unicode"):
            unicode_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to analyze...",
                lines=1
            )

            gr.Markdown("### Normalization Comparison")
            norm_df = gr.Dataframe(interactive=False)

            gr.Markdown("### Unicode Details")
            unicode_df = gr.Dataframe(interactive=False)

        # Special Tokens Tab
        with gr.Tab("Special Tokens"):
            gr.Markdown("### Standard Special Tokens")
            standard_df = gr.Dataframe(interactive=False)

            gr.Markdown("### Additional Special Tokens")
            additional_df = gr.Dataframe(interactive=False)

    # 事件绑定
    def update_models(category):
        models = get_model_choices(category)
        return gr.update(choices=models, value=models[0] if models else None)

    category_dropdown.change(
        fn=update_models,
        inputs=[category_dropdown],
        outputs=[model_dropdown]
    )

    encode_inputs = [category_dropdown, model_dropdown, encode_input, show_ids_cb, add_special_cb]
    encode_outputs = [tokens_html, stat_tokens, stat_chars, stat_chars_per_token, stat_bytes_per_token, detail_df]

    encode_input.change(fn=encode_text, inputs=encode_inputs, outputs=encode_outputs)
    show_ids_cb.change(fn=encode_text, inputs=encode_inputs, outputs=encode_outputs)
    add_special_cb.change(fn=encode_text, inputs=encode_inputs, outputs=encode_outputs)

    decode_input.change(
        fn=decode_ids,
        inputs=[category_dropdown, model_dropdown, decode_input],
        outputs=[decode_result, decode_details]
    )

    byte_input.change(
        fn=analyze_bytes,
        inputs=[category_dropdown, model_dropdown, byte_input],
        outputs=[byte_tokens_html, byte_total, byte_fallback, byte_bpe, byte_special, byte_fallback_df]
    )

    unicode_input.change(
        fn=analyze_unicode,
        inputs=[unicode_input],
        outputs=[norm_df, unicode_df]
    )

    def on_model_change_all(category, model_name):
        """模型变化时更新所有相关内容"""
        model_id = get_model_id(category, model_name)
        id_text = f"**Tokenizer**: `{model_id}`" if model_id else ""
        info = get_model_info(category, model_name)
        special_standard, special_additional = get_special_tokens(category, model_name)
        return id_text, info, special_standard, special_additional

    model_dropdown.change(
        fn=on_model_change_all,
        inputs=[category_dropdown, model_dropdown],
        outputs=[model_id_text, model_info_md, standard_df, additional_df]
    )

    def on_load():
        """页面加载时计算默认值"""
        # 只加载模型信息和特殊 token（这些不需要用户输入）
        id_text, info, special_standard, special_additional = on_model_change_all(
            initial_category, initial_models[0] if initial_models else None
        )
        # 其他 tab 没有默认输入，所以不计算默认结果
        return (
            id_text, info, special_standard, special_additional
        )

    return {
        'load_fn': on_load,
        'load_outputs': [
            model_id_text, model_info_md, standard_df, additional_df
        ]
    }
