# Learn Omni 伴随实验

这里存放网页课程的可运行代码。60 课各有一个标准库实现的 CPU 机制实验，源码位于
`src/learn_omni_experiments/lessons/`。它们用于核对张量形状、delay schedule、
路由、loss 和时间边界，不下载模型，也不需要 GPU。

这些实验不能证明模型质量。正式 GPU 实验仍需使用课程中指定的模型、数据、硬件和
对照组，并单独记录结果。

`lesson01/` 保存的是 128 样本过拟合实验的上游补丁材料，不属于 CPU runner，
也不会被 `run.py` 自动应用。

要求 Python 3.10 或更高版本。先运行一课：

```bash
python3 run.py run 01
```

```bash
python3 run.py check 01
```

运行并检查全部 60 课：

```bash
python3 run.py verify-all
```

运行代码库测试：

```bash
python3 -m unittest discover -s tests -v
```

结果写入 `artifacts/lessonXX/result.json`。面向初学者的完整操作顺序以课程网页中的
“先把本课代码跑通”为准。
