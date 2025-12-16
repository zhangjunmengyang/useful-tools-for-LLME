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
        return "请输入模型名称", None, None
    
    if not selected_dtypes:
        return "请至少选择一种精度类型", None, None
    
    try:
        progress(0.3, desc=f"正在加载模型 {model_name}...")
        token = access_token if access_token else None
        model = get_model(model_name, library, token)
        
        progress(0.7, desc="计算显存...")
        data, stages = calculate_memory_detailed(model, selected_dtypes)
        
        # 主表格
        df_main = pd.DataFrame(data)
        df_main.columns = ["精度", "最大层/残差组", "模型总大小", "训练峰值显存 (Adam)"]
        
        # 训练阶段表格
        training_stages_data = format_training_stages(stages, selected_dtypes)
        df_stages = None
        if training_stages_data:
            df_stages = pd.DataFrame(training_stages_data)
            df_stages.columns = ["精度", "模型加载", "梯度计算", "反向传播", "优化器更新"]
        
        result_msg = f"模型 `{model_name}` 的显存需求计算完成"
        
        return result_msg, df_main, df_stages
        
    except Exception as e:
        error_msg = get_model_error_message(e, model_name)
        return error_msg, None, None


def render():
    """渲染显存估算页面"""
    
    gr.Markdown("## 显存估算")
    
    # 默认值
    default_model = "bert-base-cased"
    default_library = "auto"
    default_dtypes = ["float32"]
    
    # 输入区域
    with gr.Group():
        with gr.Row():
            model_name = gr.Textbox(
                label="模型名称或 URL",
                value=default_model,
                placeholder="例如: bert-base-cased, meta-llama/Llama-2-7b-hf"
            )
            library = gr.Dropdown(
                label="模型库",
                choices=LIBRARY_OPTIONS,
                value=default_library
            )
        
        with gr.Row():
            selected_dtypes = gr.CheckboxGroup(
                label="选择精度类型",
                choices=DTYPE_OPTIONS,
                value=default_dtypes
            )
            access_token = gr.Textbox(
                label="API Token (可选)",
                type="password",
                placeholder="用于访问私有模型"
            )
    
    # 结果区域
    result_status = gr.Markdown("")
    
    main_table = gr.Dataframe(label="显存需求")
    
    with gr.Accordion("训练各阶段详情", open=False):
        stages_table = gr.Dataframe(label="训练阶段详情")
    
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
