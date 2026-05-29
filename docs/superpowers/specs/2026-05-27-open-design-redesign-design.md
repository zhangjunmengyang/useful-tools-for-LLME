# Open Design Redesign Design

## Goal

把 LLM Tools Workbench 从旧的 HuggingFace 橙色视觉壳重构为基于 `nexu-io/open-design` 的专业 LLM 工程工作台界面。

## Source Of Truth

- 主视觉系统：`nexu-io/open-design/design-systems/openai`
- 工具密度校正：`nexu-io/open-design/design-systems/default`
- 本项目约束：产品 UI 英文化，注释和 docstring 保持中文，禁止 emoji，保留 Gradio 模块自治。

## Design Direction

采用 OpenAI design system 的 white canvas、near-monochrome ink、`#10a37f` 单一品牌强调色、hairline border、12px/16px radius、Inter/system sans、JetBrains Mono fallback。页面不做营销化 hero，不引入装饰性渐变，不使用暗色 dashboard system。

## Architecture

新增 `workbench_theme.py` 作为共享主题层，集中管理 Open Design token、Gradio theme、全局 CSS、应用 header HTML 和 Plotly 默认模板。`app_gradio.py` 只负责路由和页面组合，不再持有大段视觉 CSS。

## Scope

- 重构全局 header、tabs、buttons、inputs、cards、dataframes、markdown、stat cards、token blocks。
- 将 Gradio `theme` 和 `css` 传入 `gr.Blocks`，避免在 `launch()` 中传递不稳定参数。
- 移除无条件 `inference_lab` 导入，只有目录存在时才显示该 Lab。
- 对 Eval Lab 入口页做首轮 UI 英文化和 emoji 清理，避免最明显违背 AGENTS 规则的页面继续拖累整体质感。

## Validation

- `python -m unittest tests.test_open_design_redesign -v`
- `python -m compileall app_gradio.py workbench_theme.py eval_lab`
- 如果依赖可用，启动 `python app_gradio.py` 并用浏览器检查 `http://localhost:7860`。
