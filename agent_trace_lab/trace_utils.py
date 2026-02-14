"""
Agent Trace 工具函数

提供 Agent 轨迹的数据结构、解析、统计和分析功能
"""

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class TraceEvent:
    """Agent trace 事件数据类"""
    timestamp: str                    # ISO 格式时间戳
    event_type: str                  # 事件类型: thought, action, observation, error
    agent_name: str                  # Agent 名称
    action: str                      # 动作名称 (对于 action 类型)
    input: str                       # 输入内容
    output: str                      # 输出内容
    duration_ms: float              # 执行时长 (毫秒)
    metadata: Dict[str, Any]        # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceEvent':
        """从字典创建"""
        return cls(**data)


def generate_react_trace(agent_name: str = "ReAct Agent", 
                        task: str = "Calculate the result of 15 * 23 + sqrt(144)",
                        num_steps: int = None) -> List[TraceEvent]:
    """
    生成模拟的 ReAct Agent trace

    Args:
        agent_name: Agent 名称
        task: 任务描述
        num_steps: 步骤数（None 则自动生成合理数量）

    Returns:
        TraceEvent 列表
    """
    events = []
    base_time = datetime.now()
    
    # 预定义工具和响应
    tools = {
        "search": {
            "duration_range": (800, 2000),
            "success_rate": 0.9,
            "responses": [
                "Found 3 relevant results about mathematical calculations",
                "Search completed with 5 results",
                "Mathematical operation search returned helpful information"
            ]
        },
        "calculator": {
            "duration_range": (200, 500),
            "success_rate": 0.95,
            "responses": [
                "15 * 23 = 345",
                "sqrt(144) = 12",
                "345 + 12 = 357"
            ]
        },
        "code_executor": {
            "duration_range": (1000, 3000),
            "success_rate": 0.85,
            "responses": [
                "Python code executed successfully",
                "import math\nresult = 15 * 23 + math.sqrt(144)\nprint(result)  # Output: 357.0",
                "Code execution completed"
            ]
        }
    }
    
    if num_steps is None:
        num_steps = random.randint(4, 8)
    
    step_idx = 0
    current_time = base_time
    
    # Initial thought
    thought_duration = random.uniform(500, 1500)
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="thought",
        agent_name=agent_name,
        action="",
        input=task,
        output=f"I need to solve this mathematical problem: {task}. Let me break it down into steps.",
        duration_ms=thought_duration,
        metadata={"step": step_idx, "tokens_estimate": random.randint(50, 100)}
    ))
    current_time += timedelta(milliseconds=thought_duration)
    
    for i in range(num_steps - 1):
        step_idx += 1
        
        # Choose a tool
        tool_name = random.choice(list(tools.keys()))
        tool_config = tools[tool_name]
        duration = random.uniform(*tool_config["duration_range"])
        success = random.random() < tool_config["success_rate"]
        
        # Action
        if tool_name == "calculator":
            action_input = random.choice(["15 * 23", "sqrt(144)", "345 + 12"])
        elif tool_name == "search":
            action_input = f"mathematical calculation {task.split()[0:3]}"
        else:  # code_executor
            action_input = f"import math\nresult = {task}\nprint(result)"
        
        events.append(TraceEvent(
            timestamp=current_time.isoformat(),
            event_type="action",
            agent_name=agent_name,
            action=tool_name,
            input=action_input,
            output="",
            duration_ms=duration * 0.1,  # Action planning time
            metadata={
                "step": step_idx, 
                "tool": tool_name,
                "tokens_estimate": random.randint(20, 50)
            }
        ))
        current_time += timedelta(milliseconds=duration * 0.1)
        
        # Observation
        if success:
            output = random.choice(tool_config["responses"])
        else:
            output = f"Error: {tool_name} failed to execute properly"
            
        events.append(TraceEvent(
            timestamp=current_time.isoformat(),
            event_type="observation",
            agent_name=agent_name,
            action=tool_name,
            input=action_input,
            output=output,
            duration_ms=duration * 0.9,  # Tool execution time
            metadata={
                "step": step_idx, 
                "tool": tool_name,
                "success": success,
                "tokens_estimate": random.randint(30, 80)
            }
        ))
        current_time += timedelta(milliseconds=duration * 0.9)
        
        # Thought after observation
        thought_duration = random.uniform(300, 1000)
        if i < num_steps - 2:
            thought_text = f"Good, I got the result from {tool_name}. Let me continue with the next step."
        else:
            thought_text = "Perfect! I have all the information needed to provide the final answer: 357"
            
        events.append(TraceEvent(
            timestamp=current_time.isoformat(),
            event_type="thought",
            agent_name=agent_name,
            action="",
            input="",
            output=thought_text,
            duration_ms=thought_duration,
            metadata={"step": step_idx, "tokens_estimate": random.randint(40, 90)}
        ))
        current_time += timedelta(milliseconds=thought_duration)
    
    return events


def generate_multi_agent_trace() -> List[TraceEvent]:
    """生成多Agent协作的 trace 示例"""
    events = []
    base_time = datetime.now()
    current_time = base_time
    
    agents = ["Planner", "Researcher", "Calculator", "Summarizer"]
    task = "Research the population of Tokyo and calculate population density"
    
    # Planner starts
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="thought",
        agent_name="Planner",
        action="",
        input=task,
        output="I need to break this task down: 1) Research Tokyo population, 2) Research Tokyo area, 3) Calculate density",
        duration_ms=800,
        metadata={"step": 0, "tokens_estimate": 60}
    ))
    current_time += timedelta(milliseconds=800)
    
    # Delegate to Researcher
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="action",
        agent_name="Planner",
        action="delegate",
        input="Research Tokyo population and area",
        output="Task delegated to Researcher",
        duration_ms=200,
        metadata={"step": 1, "target_agent": "Researcher", "tokens_estimate": 30}
    ))
    current_time += timedelta(milliseconds=200)
    
    # Researcher works
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="action",
        agent_name="Researcher",
        action="search",
        input="Tokyo population 2024",
        output="Tokyo population: approximately 14 million people",
        duration_ms=2500,
        metadata={"step": 2, "tool": "search", "tokens_estimate": 80}
    ))
    current_time += timedelta(milliseconds=2500)
    
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="action",
        agent_name="Researcher",
        action="search",
        input="Tokyo area square kilometers",
        output="Tokyo area: approximately 2,194 km²",
        duration_ms=1800,
        metadata={"step": 3, "tool": "search", "tokens_estimate": 70}
    ))
    current_time += timedelta(milliseconds=1800)
    
    # Calculator works
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="action",
        agent_name="Calculator",
        action="calculate",
        input="14000000 / 2194",
        output="Population density: 6,380 people per km²",
        duration_ms=300,
        metadata={"step": 4, "tool": "calculator", "tokens_estimate": 40}
    ))
    current_time += timedelta(milliseconds=300)
    
    # Summarizer finalizes
    events.append(TraceEvent(
        timestamp=current_time.isoformat(),
        event_type="thought",
        agent_name="Summarizer",
        action="",
        input="",
        output="Tokyo has a population of 14 million across 2,194 km², resulting in a density of 6,380 people/km²",
        duration_ms=600,
        metadata={"step": 5, "tokens_estimate": 85}
    ))
    
    return events


def parse_trace_from_json(json_data: str) -> List[TraceEvent]:
    """
    从 JSON 字符串解析 trace

    Args:
        json_data: JSON 格式的 trace 数据

    Returns:
        TraceEvent 列表
    """
    try:
        data = json.loads(json_data)
        if isinstance(data, list):
            return [TraceEvent.from_dict(event) for event in data]
        else:
            raise ValueError("JSON must contain a list of trace events")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")


def calculate_trace_stats(events: List[TraceEvent]) -> Dict[str, Any]:
    """
    计算 trace 统计信息

    Args:
        events: TraceEvent 列表

    Returns:
        统计信息字典
    """
    if not events:
        return {}
    
    total_duration = sum(event.duration_ms for event in events)
    step_durations = []
    tool_calls = {}
    success_count = 0
    error_count = 0
    total_tokens = 0
    
    for event in events:
        step_durations.append(event.duration_ms)
        
        # Count tool calls
        if event.event_type == "action" and event.action:
            tool_calls[event.action] = tool_calls.get(event.action, 0) + 1
        
        # Count success/errors
        if event.metadata.get("success") == True:
            success_count += 1
        elif event.metadata.get("success") == False:
            error_count += 1
        
        # Sum tokens
        total_tokens += event.metadata.get("tokens_estimate", 0)
    
    return {
        "total_duration_ms": total_duration,
        "total_duration_sec": total_duration / 1000,
        "avg_step_duration_ms": np.mean(step_durations) if step_durations else 0,
        "median_step_duration_ms": np.median(step_durations) if step_durations else 0,
        "total_steps": len(events),
        "tool_calls": tool_calls,
        "success_rate": success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0,
        "total_tokens_estimate": total_tokens,
        "events_by_type": {
            event_type: len([e for e in events if e.event_type == event_type])
            for event_type in set(event.event_type for event in events)
        }
    }


def find_critical_path(events: List[TraceEvent], top_n: int = 5) -> List[Tuple[TraceEvent, float]]:
    """
    找出最耗时的步骤 (关键路径分析)

    Args:
        events: TraceEvent 列表
        top_n: 返回前 N 个最耗时的步骤

    Returns:
        (event, duration) 元组列表，按耗时降序排列
    """
    sorted_events = sorted(events, key=lambda x: x.duration_ms, reverse=True)
    return [(event, event.duration_ms) for event in sorted_events[:top_n]]


def detect_bottlenecks(events: List[TraceEvent], threshold_multiplier: float = 2.0) -> List[Dict[str, Any]]:
    """
    检测性能瓶颈

    Args:
        events: TraceEvent 列表
        threshold_multiplier: 异常值阈值倍数

    Returns:
        瓶颈信息列表
    """
    if not events:
        return []
    
    durations = [event.duration_ms for event in events]
    median_duration = np.median(durations)
    threshold = median_duration * threshold_multiplier
    
    bottlenecks = []
    for event in events:
        if event.duration_ms > threshold:
            bottlenecks.append({
                "event": event,
                "duration_ms": event.duration_ms,
                "slowdown_factor": event.duration_ms / median_duration,
                "suggestion": get_optimization_suggestion(event)
            })
    
    return sorted(bottlenecks, key=lambda x: x["duration_ms"], reverse=True)


def get_optimization_suggestion(event: TraceEvent) -> str:
    """为瓶颈事件生成优化建议"""
    suggestions = {
        "search": "Consider caching search results or using faster search APIs",
        "code_executor": "Optimize code or use more efficient algorithms",
        "calculator": "Use optimized math libraries or parallel computation",
        "thought": "Consider reducing reasoning complexity or using faster models"
    }
    
    if event.event_type == "action":
        return suggestions.get(event.action, "Consider optimizing this operation")
    elif event.event_type == "thought":
        return suggestions["thought"]
    else:
        return "Consider optimizing this step"


def export_trace_to_json(events: List[TraceEvent]) -> str:
    """
    导出 trace 为 JSON 格式

    Args:
        events: TraceEvent 列表

    Returns:
        JSON 字符串
    """
    return json.dumps([event.to_dict() for event in events], indent=2, ensure_ascii=False)