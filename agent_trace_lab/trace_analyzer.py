"""
Agent Trace Analyzer - 轨迹分析页面

提供 Agent 轨迹的性能分析、工具使用统计和瓶颈检测
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from typing import List, Dict, Any, Tuple

from .trace_utils import (
    TraceEvent,
    parse_trace_from_json,
    calculate_trace_stats,
    find_critical_path,
    detect_bottlenecks,
    generate_react_trace,
    generate_multi_agent_trace,
    export_trace_to_json
)


def create_performance_dashboard(events: List[TraceEvent]) -> str:
    """
    创建性能仪表盘 HTML

    Args:
        events: TraceEvent 列表

    Returns:
        HTML 字符串
    """
    if not events:
        return "<p>No trace data to analyze</p>"
    
    stats = calculate_trace_stats(events)
    
    # Calculate additional metrics
    agents = set(event.agent_name for event in events)
    unique_tools = set(event.action for event in events if event.action)
    
    html = f"""
    <div class="dashboard">
        <h3>Performance Dashboard</h3>
        <div class="metrics-grid">
            <div class="metric-card primary">
                <div class="metric-value">{stats['total_duration_sec']:.2f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
            <div class="metric-card success">
                <div class="metric-value">{stats['total_steps']}</div>
                <div class="metric-label">Total Steps</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{stats['avg_step_duration_ms']:.0f}ms</div>
                <div class="metric-label">Avg Step Duration</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{len(agents)}</div>
                <div class="metric-label">Active Agents</div>
            </div>
            <div class="metric-card secondary">
                <div class="metric-value">{len(unique_tools)}</div>
                <div class="metric-label">Tools Used</div>
            </div>
            <div class="metric-card accent">
                <div class="metric-value">{stats['success_rate']*100:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card primary">
                <div class="metric-value">{stats['total_tokens_estimate']}</div>
                <div class="metric-label">Est. Tokens</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{stats['median_step_duration_ms']:.0f}ms</div>
                <div class="metric-label">Median Duration</div>
            </div>
        </div>
        
        <div class="events-breakdown">
            <h4>Events by Type</h4>
            <div class="breakdown-grid">
    """
    
    for event_type, count in stats['events_by_type'].items():
        percentage = (count / stats['total_steps']) * 100
        html += f"""
                <div class="breakdown-item">
                    <span class="event-badge event-{event_type}">{event_type}</span>
                    <span class="count">{count} ({percentage:.1f}%)</span>
                </div>
        """
    
    html += """
            </div>
        </div>
    </div>
    
    <style>
    .dashboard {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        text-align: center;
        padding: 20px;
        border-radius: 8px;
        border: 2px solid transparent;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card.primary { background: #E3F2FD; border-color: #2196F3; }
    .metric-card.success { background: #E8F5E8; border-color: #4CAF50; }
    .metric-card.info { background: #E1F5FE; border-color: #00BCD4; }
    .metric-card.warning { background: #FFF3E0; border-color: #FF9800; }
    .metric-card.secondary { background: #F3E5F5; border-color: #9C27B0; }
    .metric-card.accent { background: #FCE4EC; border-color: #E91E63; }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .events-breakdown {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 15px;
    }
    .breakdown-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin-top: 10px;
    }
    .breakdown-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: white;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    .event-badge {
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        color: white;
    }
    .event-thought { background-color: #4CAF50; }
    .event-action { background-color: #2196F3; }
    .event-observation { background-color: #FF9800; }
    .event-error { background-color: #F44336; }
    .count {
        font-weight: 600;
        color: #495057;
    }
    </style>
    """
    
    return html


def create_tool_usage_chart(events: List[TraceEvent]) -> go.Figure:
    """
    创建工具使用分布图表

    Args:
        events: TraceEvent 列表

    Returns:
        Plotly 图表对象
    """
    if not events:
        return go.Figure()
    
    # Count tool usage
    tool_counts = {}
    for event in events:
        if event.event_type == "action" and event.action:
            tool_counts[event.action] = tool_counts.get(event.action, 0) + 1
    
    if not tool_counts:
        return go.Figure().add_annotation(
            text="No tool usage data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(tool_counts.keys()),
        values=list(tool_counts.values()),
        hole=.4,
        hovertemplate="<b>%{label}</b><br>" +
                      "Calls: %{value}<br>" +
                      "Percentage: %{percent}<br>" +
                      "<extra></extra>",
        textinfo='label+percent',
        textposition='inside',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        )
    )])
    
    fig.update_layout(
        title="Tool Usage Distribution",
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        height=400,
        margin=dict(t=60, b=60, l=60, r=120)
    )
    
    return fig


def create_duration_distribution_chart(events: List[TraceEvent]) -> go.Figure:
    """
    创建耗时分布直方图

    Args:
        events: TraceEvent 列表

    Returns:
        Plotly 图表对象
    """
    if not events:
        return go.Figure()
    
    durations = [event.duration_ms for event in events]
    
    # Calculate outliers
    q75, q25 = np.percentile(durations, [75 ,25])
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)
    outliers = [d for d in durations if d < lower_bound or d > upper_bound]
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=durations,
        nbinsx=20,
        name="Duration Distribution",
        hovertemplate="Duration: %{x:.1f}ms<br>Count: %{y}<extra></extra>",
        marker_color='rgba(33, 150, 243, 0.7)',
        marker_line=dict(color='white', width=1)
    ))
    
    # Add vertical lines for quartiles
    fig.add_vline(x=np.median(durations), line_dash="dash", line_color="red",
                  annotation_text=f"Median: {np.median(durations):.1f}ms")
    fig.add_vline(x=np.mean(durations), line_dash="dot", line_color="green", 
                  annotation_text=f"Mean: {np.mean(durations):.1f}ms")
    
    # Add outlier annotations if any
    if outliers:
        outlier_y_pos = max([len([d for d in durations if abs(d - outlier) < 50]) for outlier in outliers])
        for outlier in outliers:
            fig.add_annotation(
                x=outlier, y=outlier_y_pos * 0.8,
                text=f"⚠️ {outlier:.0f}ms",
                showarrow=True, arrowhead=2,
                bgcolor="rgba(244, 67, 54, 0.8)",
                bordercolor="white", borderwidth=1,
                font_color="white"
            )
    
    fig.update_layout(
        title="Step Duration Distribution",
        xaxis_title="Duration (milliseconds)",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
    return fig


def create_bottleneck_report(events: List[TraceEvent]) -> str:
    """
    创建瓶颈检测报告

    Args:
        events: TraceEvent 列表

    Returns:
        HTML 报告
    """
    if not events:
        return "<p>No trace data to analyze</p>"
    
    critical_path = find_critical_path(events, top_n=5)
    bottlenecks = detect_bottlenecks(events)
    
    html = """
    <div class="bottleneck-report">
        <h3>🚨 Bottleneck Analysis</h3>
    """
    
    if bottlenecks:
        html += """
        <div class="bottlenecks-section">
            <h4>Detected Bottlenecks</h4>
            <div class="bottlenecks-list">
        """
        
        for i, bottleneck in enumerate(bottlenecks[:5]):  # Show top 5
            event = bottleneck['event']
            html += f"""
            <div class="bottleneck-item severity-{min(3, int(bottleneck['slowdown_factor']))}">
                <div class="bottleneck-header">
                    <span class="bottleneck-badge">#{i+1}</span>
                    <strong>{event.event_type.title()}</strong>
                    {f"({event.action})" if event.action else ""}
                    <span class="duration">{bottleneck['duration_ms']:.1f}ms</span>
                </div>
                <div class="bottleneck-details">
                    <div class="slowdown">
                        <strong>Slowdown:</strong> {bottleneck['slowdown_factor']:.1f}x slower than median
                    </div>
                    <div class="suggestion">
                        <strong>💡 Suggestion:</strong> {bottleneck['suggestion']}
                    </div>
                </div>
            </div>
            """
        html += "</div></div>"
    else:
        html += """
        <div class="no-bottlenecks">
            <p>✅ No significant bottlenecks detected. Performance appears well-balanced.</p>
        </div>
        """
    
    # Critical path analysis
    html += """
    <div class="critical-path-section">
        <h4>🎯 Critical Path (Top 5 Slowest Steps)</h4>
        <div class="critical-path-list">
    """
    
    total_duration = sum(event.duration_ms for event in events)
    for i, (event, duration) in enumerate(critical_path):
        percentage = (duration / total_duration) * 100
        html += f"""
        <div class="critical-step">
            <div class="step-rank">#{i+1}</div>
            <div class="step-info">
                <div class="step-title">{event.event_type.title()}</div>
                <div class="step-meta">{event.agent_name} • {event.action if event.action else 'N/A'}</div>
            </div>
            <div class="step-metrics">
                <div class="duration">{duration:.1f}ms</div>
                <div class="percentage">({percentage:.1f}%)</div>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    </div>
    
    <style>
    .bottleneck-report {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    .bottlenecks-section {
        margin-bottom: 25px;
    }
    .bottlenecks-list {
        space-y: 10px;
    }
    .bottleneck-item {
        border-left: 4px solid #ddd;
        background: #f8f9fa;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .bottleneck-item.severity-1 { border-left-color: #ffc107; }
    .bottleneck-item.severity-2 { border-left-color: #fd7e14; }
    .bottleneck-item.severity-3 { border-left-color: #dc3545; }
    .bottleneck-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .bottleneck-badge {
        background: #6c757d;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: bold;
    }
    .duration {
        background: #e9ecef;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-left: auto;
    }
    .bottleneck-details {
        font-size: 13px;
        color: #6c757d;
        line-height: 1.4;
    }
    .slowdown {
        margin-bottom: 5px;
    }
    .suggestion {
        font-style: italic;
    }
    .no-bottlenecks {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 6px;
        padding: 15px;
        color: #0c5460;
    }
    .critical-path-section {
        border-top: 1px solid #dee2e6;
        padding-top: 20px;
    }
    .critical-path-list {
        space-y: 8px;
    }
    .critical-step {
        display: flex;
        align-items: center;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .step-rank {
        background: #495057;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        margin-right: 15px;
    }
    .step-info {
        flex-grow: 1;
    }
    .step-title {
        font-weight: 600;
        margin-bottom: 2px;
    }
    .step-meta {
        font-size: 12px;
        color: #6c757d;
    }
    .step-metrics {
        text-align: right;
    }
    .step-metrics .duration {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 2px;
    }
    .step-metrics .percentage {
        font-size: 12px;
        color: #6c757d;
    }
    </style>
    """
    
    return html


def load_trace_for_analysis(events_json: str) -> tuple:
    """从 JSON 加载 trace 进行分析"""
    if not events_json:
        return None, None, "<p>No trace data to analyze</p>"
    
    try:
        events_data = json.loads(events_json)
        events = [TraceEvent.from_dict(event_data) for event_data in events_data]
        
        dashboard = create_performance_dashboard(events)
        tool_usage_chart = create_tool_usage_chart(events)
        duration_chart = create_duration_distribution_chart(events)
        bottleneck_report = create_bottleneck_report(events)
        
        return tool_usage_chart, duration_chart, dashboard, bottleneck_report
        
    except Exception as e:
        error_msg = f"<p>Error analyzing trace: {str(e)}</p>"
        return None, None, error_msg, error_msg


def load_example_for_analysis(trace_type: str) -> tuple:
    """加载示例 trace 进行分析"""
    try:
        if trace_type == "react":
            events = generate_react_trace()
        elif trace_type == "multi_agent":
            events = generate_multi_agent_trace()
        else:
            events = []
        
        if not events:
            empty_msg = "<p>No trace data</p>"
            return None, None, empty_msg, empty_msg
        
        dashboard = create_performance_dashboard(events)
        tool_usage_chart = create_tool_usage_chart(events)
        duration_chart = create_duration_distribution_chart(events)
        bottleneck_report = create_bottleneck_report(events)
        
        return tool_usage_chart, duration_chart, dashboard, bottleneck_report
        
    except Exception as e:
        error_msg = f"<p>Error loading example: {str(e)}</p>"
        return None, None, error_msg, error_msg


def render() -> Dict[str, Any]:
    """渲染 Trace Analyzer 页面"""
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Load Trace for Analysis")
            
            with gr.Row():
                react_btn = gr.Button("ReAct Example", variant="secondary", size="sm")
                multi_btn = gr.Button("Multi-Agent Example", variant="secondary", size="sm")
            
            json_file = gr.File(
                label="Upload Trace JSON",
                file_count="single",
                file_types=[".json"]
            )
    
    # Performance Dashboard
    with gr.Row():
        dashboard = gr.HTML("<p>Load trace data to view performance dashboard</p>")
    
    # Charts Row
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Tool Usage Distribution")
            tool_usage_chart = gr.Plot(label="Tool Usage")
        
        with gr.Column(scale=1):
            gr.Markdown("### Duration Distribution")
            duration_chart = gr.Plot(label="Duration Analysis")
    
    # Bottleneck Report
    with gr.Row():
        bottleneck_report = gr.HTML("<p>Load trace data to view bottleneck analysis</p>")
    
    # Event handlers
    react_btn.click(
        fn=lambda: load_example_for_analysis("react"),
        outputs=[tool_usage_chart, duration_chart, dashboard, bottleneck_report]
    )
    
    multi_btn.click(
        fn=lambda: load_example_for_analysis("multi_agent"),
        outputs=[tool_usage_chart, duration_chart, dashboard, bottleneck_report]
    )
    
    json_file.change(
        fn=lambda file: load_trace_for_analysis(file.decode() if file else ""),
        inputs=[json_file],
        outputs=[tool_usage_chart, duration_chart, dashboard, bottleneck_report]
    )
    
    return {
        'load_fn': lambda: load_example_for_analysis("react"),
        'load_outputs': [tool_usage_chart, duration_chart, dashboard, bottleneck_report]
    }