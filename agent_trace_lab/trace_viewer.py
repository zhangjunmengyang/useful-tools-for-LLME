"""
Agent Trace Viewer - 轨迹可视化页面

提供 Agent 轨迹的时间线视图、步骤详情和流程图
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import List, Dict, Any, Optional

from .trace_utils import (
    TraceEvent, 
    generate_react_trace, 
    generate_multi_agent_trace,
    parse_trace_from_json,
    export_trace_to_json
)


def create_timeline_chart(events: List[TraceEvent]) -> go.Figure:
    """
    创建 Agent 执行时间线甘特图

    Args:
        events: TraceEvent 列表

    Returns:
        Plotly 图表对象
    """
    if not events:
        return go.Figure()
    
    # 准备数据
    chart_data = []
    colors = {
        'thought': '#4CAF50',      # Green
        'action': '#2196F3',       # Blue  
        'observation': '#FF9800',  # Orange
        'error': '#F44336'         # Red
    }
    
    start_time = datetime.fromisoformat(events[0].timestamp.replace('Z', '+00:00'))
    
    for i, event in enumerate(events):
        event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
        start_offset = (event_time - start_time).total_seconds() * 1000  # Convert to ms
        end_offset = start_offset + event.duration_ms
        
        # Create hover text
        hover_text = f"""
        <b>{event.event_type.title()}</b><br>
        Agent: {event.agent_name}<br>
        {'Action: ' + event.action + '<br>' if event.action else ''}
        Duration: {event.duration_ms:.1f}ms<br>
        Input: {event.input[:50] + '...' if len(event.input) > 50 else event.input}<br>
        Output: {event.output[:50] + '...' if len(event.output) > 50 else event.output}
        """
        
        chart_data.append({
            'Task': f"Step {i+1} ({event.event_type})",
            'Start': start_offset,
            'Finish': end_offset,
            'Resource': event.agent_name,
            'Description': event.action if event.action else event.event_type,
            'Color': colors.get(event.event_type, '#9E9E9E'),
            'Hover': hover_text,
            'EventType': event.event_type,
            'Index': i
        })
    
    # Create Gantt chart
    fig = go.Figure()
    
    # Group by agent
    agents = list(set(item['Resource'] for item in chart_data))
    agent_y_pos = {agent: i for i, agent in enumerate(agents)}
    
    for item in chart_data:
        fig.add_trace(go.Bar(
            x=[item['Finish'] - item['Start']],
            y=[agent_y_pos[item['Resource']]],
            base=[item['Start']],
            orientation='h',
            name=item['Task'],
            marker_color=item['Color'],
            text=item['Description'],
            textposition='inside',
            hovertemplate=item['Hover'] + '<extra></extra>',
            customdata=[item['Index']],  # Store index for selection
            showlegend=False
        ))
    
    fig.update_layout(
        title="Agent Execution Timeline",
        xaxis_title="Time (milliseconds)",
        yaxis_title="Agent",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(agents))),
            ticktext=agents
        ),
        height=max(400, len(agents) * 80),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
    return fig


def create_flow_diagram(events: List[TraceEvent]) -> str:
    """
    创建 Agent 流程图 (Mermaid 格式)

    Args:
        events: TraceEvent 列表

    Returns:
        Mermaid diagram 字符串
    """
    if not events:
        return "graph TD\n    A[No trace data]"
    
    mermaid_lines = ["graph TD"]
    
    # Group consecutive events by type for cleaner flow
    flow_groups = []
    current_group = {"type": events[0].event_type, "count": 1, "start_idx": 0}
    
    for i in range(1, len(events)):
        if events[i].event_type == current_group["type"]:
            current_group["count"] += 1
        else:
            flow_groups.append(current_group)
            current_group = {"type": events[i].event_type, "count": 1, "start_idx": i}
    flow_groups.append(current_group)
    
    # Generate mermaid nodes and connections
    for i, group in enumerate(flow_groups):
        node_id = f"node{i}"
        
        if group["type"] == "thought":
            shape = f"{node_id}[Thinking]"
            style = f"    {node_id} --> "
        elif group["type"] == "action":
            action_name = events[group["start_idx"]].action
            shape = f"{node_id}[{action_name}]"
            style = f"    {node_id} --> "
        elif group["type"] == "observation":
            shape = f"{node_id}[Observation]"
            style = f"    {node_id} --> "
        else:
            shape = f"{node_id}[Error]"
            style = f"    {node_id} --> "
        
        mermaid_lines.append(f"    {shape}")
        
        # Add connection to next node
        if i < len(flow_groups) - 1:
            next_node_id = f"node{i+1}"
            mermaid_lines.append(f"    {node_id} --> {next_node_id}")
    
    # Add final completion node
    if flow_groups:
        last_node = f"node{len(flow_groups)-1}"
        mermaid_lines.append(f"    {last_node} --> Complete[Complete]")
    
    # Add styling
    mermaid_lines.extend([
        "    classDef thinking fill:#E8F5E8,stroke:#4CAF50",
        "    classDef action fill:#E3F2FD,stroke:#2196F3", 
        "    classDef observation fill:#FFF3E0,stroke:#FF9800",
        "    classDef error fill:#FFEBEE,stroke:#F44336",
        "    classDef complete fill:#F1F8E9,stroke:#689F38"
    ])
    
    return "\n".join(mermaid_lines)


def get_step_details(events: List[TraceEvent], step_index: int) -> Dict[str, Any]:
    """
    获取指定步骤的详细信息

    Args:
        events: TraceEvent 列表
        step_index: 步骤索引

    Returns:
        步骤详情字典
    """
    if not events or step_index < 0 or step_index >= len(events):
        return {}
    
    event = events[step_index]
    return {
        "step_number": step_index + 1,
        "timestamp": event.timestamp,
        "event_type": event.event_type,
        "agent_name": event.agent_name,
        "action": event.action,
        "input": event.input,
        "output": event.output,
        "duration_ms": event.duration_ms,
        "metadata": event.metadata
    }


def format_step_details_html(details: Dict[str, Any]) -> str:
    """将步骤详情格式化为 HTML"""
    if not details:
        return "<p>No step selected</p>"
    
    metadata_items = ""
    for key, value in details["metadata"].items():
        metadata_items += f"<li><strong>{key}:</strong> {value}</li>"
    
    html = f"""
    <div class="step-details">
        <h3>Step {details['step_number']} Details</h3>
        <div class="detail-grid">
            <div class="detail-item">
                <strong>Event Type:</strong> 
                <span class="badge badge-{details['event_type']}">{details['event_type'].title()}</span>
            </div>
            <div class="detail-item">
                <strong>Agent:</strong> {details['agent_name']}
            </div>
            <div class="detail-item">
                <strong>Timestamp:</strong> {details['timestamp']}
            </div>
            <div class="detail-item">
                <strong>Duration:</strong> {details['duration_ms']:.1f}ms
            </div>
            {f'<div class="detail-item"><strong>Action:</strong> {details["action"]}</div>' if details["action"] else ''}
        </div>
        
        <div class="detail-section">
            <h4>Input</h4>
            <div class="code-block">{details['input'] if details['input'] else '<em>No input</em>'}</div>
        </div>
        
        <div class="detail-section">
            <h4>Output</h4>
            <div class="code-block">{details['output'] if details['output'] else '<em>No output</em>'}</div>
        </div>
        
        <div class="detail-section">
            <h4>Metadata</h4>
            <ul class="metadata-list">{metadata_items}</ul>
        </div>
    </div>
    
    <style>
    .step-details {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 20px;
    }
    .detail-item {
        background: white;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        color: white;
    }
    .badge-thought { background-color: #4CAF50; }
    .badge-action { background-color: #2196F3; }
    .badge-observation { background-color: #FF9800; }
    .badge-error { background-color: #F44336; }
    .detail-section {
        margin-bottom: 15px;
    }
    .code-block {
        background: #f1f3f4;
        border: 1px solid #d0d7de;
        border-radius: 4px;
        padding: 12px;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 13px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .metadata-list {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 10px;
        margin: 0;
    }
    </style>
    """
    return html


def handle_trace_selection(events_json: str, selected_data: Dict) -> str:
    """处理时间线图表的点击事件"""
    if not events_json or not selected_data:
        return "<p>Click on a step in the timeline to view details</p>"
    
    try:
        events_data = json.loads(events_json)
        events = [TraceEvent.from_dict(event_data) for event_data in events_data]
        
        # Extract step index from selected data
        if 'points' in selected_data and len(selected_data['points']) > 0:
            point = selected_data['points'][0]
            if 'customdata' in point and isinstance(point['customdata'], (list, tuple)):
                step_index = point['customdata'][0]
                details = get_step_details(events, step_index)
                return format_step_details_html(details)
        
        return "<p>Click on a step in the timeline to view details</p>"
    except Exception as e:
        return f"<p>Error loading step details: {str(e)}</p>"


def load_example_trace(trace_type: str) -> tuple:
    """加载示例 trace"""
    try:
        if trace_type == "react":
            events = generate_react_trace()
        elif trace_type == "multi_agent":
            events = generate_multi_agent_trace()
        else:
            events = []
        
        if not events:
            return None, "", "<p>No trace data</p>"
        
        # Create visualizations
        timeline_fig = create_timeline_chart(events)
        flow_diagram = create_flow_diagram(events)
        events_json = export_trace_to_json(events)
        
        return timeline_fig, events_json, "<p>Select a step from the timeline to view details</p>"
    
    except Exception as e:
        return None, "", f"<p>Error loading example: {str(e)}</p>"


def load_trace_from_json(json_text: str) -> tuple:
    """从上传的 JSON 加载 trace"""
    if not json_text.strip():
        return None, "", "<p>Please upload a JSON file</p>"
    
    try:
        events = parse_trace_from_json(json_text)
        timeline_fig = create_timeline_chart(events)
        flow_diagram = create_flow_diagram(events)
        events_json = export_trace_to_json(events)
        
        return timeline_fig, events_json, "<p>JSON trace loaded successfully. Select a step to view details</p>"
    
    except Exception as e:
        return None, "", f"<p>Error parsing JSON: {str(e)}</p>"


def render() -> Dict[str, Any]:
    """渲染 Trace Viewer 页面"""

    gr.HTML("""
    <div class="workbench-page-hero">
      <h1>Trace Viewer</h1>
      <p>Load an agent trace, inspect execution timing, and review step-level observations.</p>
    </div>
    """)

    with gr.Row(elem_classes=["workbench-tool-shell"]):
        with gr.Column(scale=1, elem_classes=["workbench-control-panel"]):
            gr.HTML("""
            <p class="workbench-panel-title">Trace Source</p>
            <p class="workbench-panel-copy">Start from a built-in example or upload a JSON trace from an agent run.</p>
            """)

            with gr.Row():
                react_btn = gr.Button("ReAct Example", variant="secondary", size="sm")
                multi_btn = gr.Button("Multi-Agent Example", variant="secondary", size="sm")
            
            json_file = gr.File(
                label="Upload Trace JSON",
                file_count="single",
                file_types=[".json"]
            )
            
            # Hidden state to store events data
            events_state = gr.Textbox(visible=False)

        with gr.Column(scale=3, elem_classes=["workbench-output-panel"]):
            with gr.Row():
                with gr.Column(scale=2, elem_classes=["plot-frame"]):
                    gr.Markdown("### Timeline View")
                    timeline_plot = gr.Plot(label="Agent Execution Timeline")

                with gr.Column(scale=1, elem_classes=["workbench-detail-panel"]):
                    gr.Markdown("### Step Details")
                    step_details = gr.HTML("<p>Select a step from the timeline to view details</p>")

            with gr.Column(elem_classes=["code-surface"]):
                gr.Markdown("### Agent Flow Diagram")
                flow_diagram = gr.Textbox(
                    label="Mermaid Flow Diagram",
                    lines=15,
                    value="graph TD\n    A[Click example buttons to load trace data]",
                    interactive=False
                )
    
    # Event handlers
    react_btn.click(
        fn=lambda: load_example_trace("react"),
        outputs=[timeline_plot, events_state, step_details]
    )
    
    multi_btn.click(
        fn=lambda: load_example_trace("multi_agent"), 
        outputs=[timeline_plot, events_state, step_details]
    )
    
    json_file.change(
        fn=lambda file: load_trace_from_json(file.decode() if file else ""),
        inputs=[json_file],
        outputs=[timeline_plot, events_state, step_details]
    )
    
    # Gradio 6 removed Plot.select; keep app construction compatible.
    if hasattr(timeline_plot, "select"):
        timeline_plot.select(
            fn=handle_trace_selection,
            inputs=[events_state, timeline_plot],
            outputs=[step_details]
        )
    
    return {
        'load_fn': lambda: load_example_trace("react"),
        'load_outputs': [timeline_plot, events_state, step_details]
    }
