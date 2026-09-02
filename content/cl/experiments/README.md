# Learn CL 伴随实验

24 课各有一个 CPU 机制实验。不下载模型，不需要 GPU。

```bash
python3 run.py run 01
```

```bash
python3 run.py verify-all
```

结果写入 `artifacts/lessonNN/result.json`。24 课机制实验均应 `PASS`；它们钉的是缩小机制，不能代替锚定仓库上的 GPU 复现。

第 24 课之后把四条写入通道放到同一条日历流上：

```bash
python3 run.py capstone
```

说明见仓库根目录 `CAPSTONE.md`。

额外实验（28 个：卸库、巩固、毕业卸库、生成回放、主动遗忘、过关才出日记）。接法见仓库根目录 `AGENT_MEMORY.md`：

```bash
python3 run.py extra run all
```

GPU 开源仓库配方（不替代 CPU 机制）：

```bash
python3 run.py gpu list
```

```bash
python3 run.py gpu print razor-mnist
```

```bash
python3 run.py gpu smoke
```

教程见仓库根目录 `GPU.md`。`smoke` 不下载模型；没装 torch 会跳过。
