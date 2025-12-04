# TokenLab - 分词实验室

一站式 LLM 分词器可视化、调试与效率分析工作台。

## 功能模块

### 1. Playground (交互式编解码)
- 支持 HuggingFace 模型动态加载
- 彩虹分词可视化
- Token ID 双向互查
- 压缩率统计

### 2. Arena (分词竞技场)
- 多模型分词效果对比
- 压缩效率指标分析 (Token Count, CPT)
- 可视化对比图表

### 3. X-Ray (字节级透视)
- Byte Fallback 机制分析
- Unicode 规范化检测 (NFC/NFD/NFKC/NFKD)
- 特殊 Token 映射查看

### 4. Chat Builder (对话模版调试)
- Chat Template 渲染预览
- 特殊 Token 高亮显示
- SFT 数据格式校验

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

## 支持的分词算法

- BPE (Byte-Pair Encoding)
- WordPiece
- Unigram / SentencePiece

## 预设模型

- gpt2
- bert-base-uncased
- bert-base-chinese
- meta-llama/Llama-2-7b-hf
- meta-llama/Meta-Llama-3-8B
- Qwen/Qwen2-7B
- mistralai/Mistral-7B-v0.1
- 等

## 技术栈

- Python 3.8+
- Streamlit
- Transformers (HuggingFace)
- Plotly

## License

MIT
