"""
ModelLab - 显存估算
估算模型推理和训练所需的显存大小
"""

import gradio as gr
import pandas as pd
from accelerate.utils import convert_bytes

from model_lab.model_utils import (
    DTYPE_OPTIONS,
    LIBRARY_OPTIONS,
    get_model,
    calculate_memory_detailed,
    format_training_stages,
    get_model_error_message,
)


def estimate_memory(model_name: str, library: str, selected_dtypes: list, access_token: str, progress=gr.Progress()):
    """估算模型显存"""
    if not model_name:
        return "Please enter model name", None, None

    if not selected_dtypes:
        return "Please select at least one precision type", None, None

    try:
        progress(0.3, desc=f"Loading model {model_name}...")
        token = access_token if access_token else None
        model = get_model(model_name, library, token)

        progress(0.7, desc="Calculating memory...")
        data, stages = calculate_memory_detailed(model, selected_dtypes)

        # 主表格
        df_main = pd.DataFrame(data)
        df_main.columns = ["Precision", "Max Layer/Residual Group", "Model Total Size", "Training Peak Memory (Adam)"]

        # 训练阶段表格
        training_stages_data = format_training_stages(stages, selected_dtypes)
        df_stages = None
        if training_stages_data:
            df_stages = pd.DataFrame(training_stages_data)
            df_stages.columns = ["Precision", "Model Loading", "Gradient Calculation", "Backward Pass", "Optimizer Update"]

        result_msg = f"Memory requirements for `{model_name}` calculated"

        return result_msg, df_main, df_stages
        
    except Exception as e:
        error_msg = get_model_error_message(e, model_name)
        return error_msg, None, None


def render():
    """渲染显存估算页面"""

    # 默认值
    default_model = "bert-base-cased"
    default_library = "auto"
    default_dtypes = ["float32"]
    
    # 输入区域
    with gr.Group():
        with gr.Row():
            model_name = gr.Textbox(
                label="Model Name or URL",
                value=default_model,
                placeholder="e.g., bert-base-cased, meta-llama/Llama-2-7b-hf"
            )
            library = gr.Dropdown(
                label="Model Library",
                choices=LIBRARY_OPTIONS,
                value=default_library
            )

        with gr.Row():
            selected_dtypes = gr.CheckboxGroup(
                label="Select Precision Types",
                choices=DTYPE_OPTIONS,
                value=default_dtypes
            )
            access_token = gr.Textbox(
                label="API Token (Optional)",
                type="password",
                placeholder="For accessing private models"
            )

    # 结果区域
    result_status = gr.Markdown("")

    main_table = gr.Dataframe(label="Memory Requirements")

    with gr.Accordion("Training Stage Details", open=False):
        stages_table = gr.Dataframe(label="Training Stage Details")
    
    # 参数变化自动触发计算
    inputs = [model_name, library, selected_dtypes, access_token]
    outputs = [result_status, main_table, stages_table]
    
    # 模型名称使用 submit 事件（按回车触发），避免每次输入都触发
    model_name.submit(fn=estimate_memory, inputs=inputs, outputs=outputs)
    library.change(fn=estimate_memory, inputs=inputs, outputs=outputs)
    selected_dtypes.change(fn=estimate_memory, inputs=inputs, outputs=outputs)
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        return estimate_memory(default_model, default_library, default_dtypes, "")
    
    # 返回 load 事件需要的信息供主 app 调用
    return {
        'load_fn': on_load,
        'load_outputs': outputs
    }
