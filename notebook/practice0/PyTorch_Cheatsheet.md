# PyTorch 快速参考手册

简洁的 PyTorch 语法速查表，每个函数一个示例

## 1. 张量创建

```python
# 从数据创建
torch.tensor([1, 2, 3])  # 从列表创建
torch.from_numpy(np.array([1, 2, 3]))  # 从 numpy 创建

# 特殊张量
torch.zeros(2, 3)  # 全0
torch.ones(2, 3)  # 全1
torch.eye(3)  # 单位矩阵
torch.empty(2, 3)  # 未初始化
torch.full((2, 3), 7)  # 填充指定值
torch.arange(0, 10, 2)  # 等差序列: [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)  # 线性空间: [0, 0.25, 0.5, 0.75, 1.0]

# 随机张量
torch.rand(2, 3)  # 均匀分布 [0, 1)
torch.randn(2, 3)  # 标准正态分布
torch.randint(0, 10, (2, 3))  # 随机整数 [0, 10)
torch.randperm(5)  # 随机排列: [3, 1, 4, 0, 2]

# like 操作（保持形状和类型）
x = torch.randn(2, 3)
torch.zeros_like(x)
torch.ones_like(x)
torch.rand_like(x)
```

## 2. 张量属性

```python
x = torch.randn(2, 3, 4)

x.shape  # 或 x.size(): torch.Size([2, 3, 4])
x.dim()  # 维度数: 3
x.numel()  # 元素总数: 24
x.dtype  # 数据类型: torch.float32
x.device  # 设备: cpu 或 cuda
x.requires_grad  # 是否需要梯度
```

## 3. 数据类型与设备

```python
# 类型转换
x = torch.randn(2, 3)
x.int()  # 转为整型
x.long()  # 转为长整型
x.float()  # 转为浮点型
x.double()  # 转为双精度
x.half()  # 转为半精度
x.bool()  # 转为布尔型
x.to(torch.int32)  # 通用转换

# 设备转换
x.cuda()  # 移到 GPU
x.cpu()  # 移到 CPU
x.to('cuda:0')  # 移到指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x.to(device)
```

## 4. 索引与切片

```python
x = torch.randn(4, 5, 6)

# 基本索引
x[0]  # 第一个元素
x[:2]  # 前两个
x[1, :, 3]  # 第2个batch，所有行，第4列
x[..., 0]  # 最后一维的第0个，等价于 x[:, :, 0]

# 布尔索引
mask = x > 0
x[mask]  # 选择所有正值

# 高级索引
torch.index_select(x, dim=0, index=torch.tensor([0, 2]))  # 选择第0和第2行
torch.masked_select(x, mask)  # 根据 mask 选择
torch.gather(x, dim=1, index=indices)  # 按索引收集
torch.take(x, torch.tensor([0, 10, 20]))  # 按展平后的索引
```

## 5. 形状操作

```python
x = torch.randn(2, 3, 4)

# 改变形状
x.view(2, 12)  # 重塑，要求内存连续
x.reshape(2, 12)  # 重塑，自动处理内存
x.view(-1)  # 展平: [24]
x.flatten()  # 展平所有维度
x.flatten(1)  # 从第1维开始展平: [2, 12]

# 增减维度
x.unsqueeze(0)  # 在第0维增加: [1, 2, 3, 4]
x.unsqueeze(-1)  # 在最后增加: [2, 3, 4, 1]
x.squeeze()  # 移除所有大小为1的维度
x.squeeze(1)  # 移除指定维度（如果大小为1）

# 转置
x.t()  # 2D 转置
x.transpose(0, 1)  # 交换维度: [3, 2, 4]
x.permute(2, 0, 1)  # 重排所有维度: [4, 2, 3]
x.transpose(0, 1).contiguous()  # 转置后变连续

# 拼接与分割
torch.cat([x, x], dim=0)  # 沿维度拼接
torch.stack([x, x], dim=0)  # 新建维度拼接: [2, 2, 3, 4]
torch.chunk(x, 2, dim=0)  # 平均分割成2份
torch.split(x, [1, 1], dim=0)  # 按指定大小分割

# 扩展与重复
x = torch.randn(1, 3)
x.expand(4, 3)  # 扩展维度（不复制数据）
x.repeat(4, 1)  # 重复（复制数据）: [4, 3]
x.repeat_interleave(3, dim=0)  # 每个元素重复3次
```

## 6. 数学运算

```python
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# 基本运算
x + y  # 或 torch.add(x, y)
x - y
x * y  # 逐元素乘法
x / y
x ** 2  # 幂运算
x % 2  # 取模
torch.div(x, y, rounding_mode='floor')  # 整除

# 原地操作（in-place）
x.add_(1)  # x = x + 1
x.mul_(2)  # x = x * 2

# 矩阵运算
torch.mm(x, y.t())  # 矩阵乘法 (2D)
torch.matmul(x, y.t())  # 通用矩阵乘法（支持广播）
x @ y.t()  # 矩阵乘法运算符
torch.bmm(x.unsqueeze(0), y.unsqueeze(0).transpose(1, 2))  # batch 矩阵乘法
torch.einsum('ij,jk->ik', x, y.t())  # Einstein 求和

# 数学函数
torch.abs(x)  # 绝对值
torch.sqrt(x.abs())  # 平方根
torch.exp(x)  # 指数
torch.log(x.abs())  # 对数
torch.sin(x)  # 三角函数
torch.sigmoid(x)  # sigmoid
torch.tanh(x)  # tanh
torch.clamp(x, min=0, max=1)  # 裁剪到范围
torch.clip(x, 0, 1)  # 同上
torch.round(x)  # 四舍五入
torch.floor(x)  # 向下取整
torch.ceil(x)  # 向上取整
```

## 7. 归约操作

```python
x = torch.randn(2, 3, 4)

# 统计
x.sum()  # 所有元素求和
x.sum(dim=1)  # 沿指定维度求和
x.sum(dim=1, keepdim=True)  # 保持维度
x.mean()  # 均值
x.std()  # 标准差
x.var()  # 方差
x.max()  # 最大值
x.min()  # 最小值
x.max(dim=1)  # 返回 (values, indices)
x.argmax(dim=1)  # 最大值索引
x.argmin(dim=1)  # 最小值索引

# 其他
x.prod()  # 乘积
x.cumsum(dim=0)  # 累积和
x.cumprod(dim=0)  # 累积乘积
torch.median(x)  # 中位数
torch.topk(x, 3, dim=1)  # top-k 值和索引
torch.sort(x, dim=1)  # 排序
```

## 8. 比较与逻辑

```python
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# 比较运算
x > 0  # 返回布尔张量
x >= y
x == y
x != y
torch.eq(x, y)  # 等于
torch.gt(x, y)  # 大于
torch.lt(x, y)  # 小于

# 逻辑运算
torch.all(x > 0)  # 所有元素满足
torch.any(x > 0)  # 任一元素满足
torch.logical_and(x > 0, x < 1)
torch.logical_or(x > 0, x < -1)
torch.logical_not(x > 0)

# 条件选择
torch.where(x > 0, x, torch.zeros_like(x))  # 条件选择
```

## 9. 神经网络层

```python
# 线性层
nn.Linear(10, 20)  # 输入10维，输出20维

# 卷积层
nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
nn.Conv3d(1, 8, kernel_size=3)

# 池化层
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d((1, 1))  # 自适应到指定大小

# 归一化层
nn.BatchNorm1d(20)
nn.BatchNorm2d(64)
nn.LayerNorm(512)  # 常用于 Transformer
nn.GroupNorm(num_groups=8, num_channels=64)
nn.InstanceNorm2d(64)

# 激活函数
nn.ReLU()
nn.LeakyReLU(0.1)
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=-1)
nn.LogSoftmax(dim=-1)

# Dropout
nn.Dropout(0.5)
nn.Dropout2d(0.3)

# Embedding
nn.Embedding(num_embeddings=1000, embedding_dim=128)  # 词表1000，维度128

# RNN 系列
nn.RNN(input_size=10, hidden_size=20, num_layers=2)
nn.LSTM(10, 20, 2, batch_first=True)
nn.GRU(10, 20, 2)

# Transformer
nn.MultiheadAttention(embed_dim=512, num_heads=8)
nn.TransformerEncoderLayer(d_model=512, nhead=8)

# 容器
nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
nn.ModuleDict({'linear': nn.Linear(10, 10), 'relu': nn.ReLU()})
```

## 10. 损失函数

```python
# 分类
nn.CrossEntropyLoss()  # 多分类（包含 softmax）
nn.NLLLoss()  # 负对数似然（需先 log_softmax）
nn.BCELoss()  # 二分类（需先 sigmoid）
nn.BCEWithLogitsLoss()  # 二分类（包含 sigmoid）

# 回归
nn.MSELoss()  # 均方误差
nn.L1Loss()  # 平均绝对误差
nn.SmoothL1Loss()  # Smooth L1（Huber Loss）

# 其他
nn.KLDivLoss()  # KL 散度
nn.CosineEmbeddingLoss()  # 余弦相似度
nn.TripletMarginLoss()  # 三元组损失
```

## 11. 优化器

```python
import torch.optim as optim

model = nn.Linear(10, 1)

# 常用优化器
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Adam + 权重衰减
optim.RMSprop(model.parameters(), lr=0.01)
optim.Adagrad(model.parameters(), lr=0.01)

# 学习率调度器
optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

# 使用示例
for epoch in range(100):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## 12. 自动微分

```python
# 开启梯度
x = torch.randn(2, 3, requires_grad=True)
y = x ** 2
z = y.mean()
z.backward()  # 反向传播
x.grad  # 查看梯度

# 梯度操作
x.grad.zero_()  # 清零梯度
torch.autograd.grad(z, x)  # 计算梯度但不保存

# 上下文管理
with torch.no_grad():  # 禁用梯度计算
    y = x * 2

with torch.set_grad_enabled(False):  # 同上
    y = x * 2

x.detach()  # 分离计算图
x.requires_grad_(False)  # 停止跟踪梯度

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

## 13. 模型操作

```python
model = nn.Linear(10, 1)

# 模式切换
model.train()  # 训练模式（启用 dropout、batch norm 更新）
model.eval()  # 评估模式（禁用 dropout、batch norm 不更新）

# 参数访问
model.parameters()  # 所有参数
model.named_parameters()  # 参数名和参数
model.state_dict()  # 参数字典
model.load_state_dict(state_dict)  # 加载参数

# 冻结与解冻
for param in model.parameters():
    param.requires_grad = False  # 冻结

# 保存与加载
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
torch.save(model, 'model_full.pth')  # 保存整个模型
model = torch.load('model_full.pth')

# 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

## 14. 权重初始化

```python
w = torch.empty(3, 5)

nn.init.uniform_(w, a=0, b=1)  # 均匀分布
nn.init.normal_(w, mean=0, std=1)  # 正态分布
nn.init.constant_(w, 0.5)  # 常数
nn.init.ones_(w)
nn.init.zeros_(w)
nn.init.xavier_uniform_(w)  # Xavier 初始化
nn.init.xavier_normal_(w)
nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')  # He 初始化
nn.init.kaiming_normal_(w)
nn.init.orthogonal_(w)  # 正交初始化
```

## 15. 函数式 API (F)

```python
x = torch.randn(2, 3, 4, 5)

# 激活函数
F.relu(x)
F.gelu(x)
F.sigmoid(x)
F.tanh(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)

# 损失函数
F.cross_entropy(logits, targets)
F.nll_loss(log_probs, targets)
F.mse_loss(pred, target)
F.binary_cross_entropy(pred, target)
F.binary_cross_entropy_with_logits(logits, target)

# 卷积和池化
F.conv2d(x, weight)
F.max_pool2d(x, kernel_size=2)
F.avg_pool2d(x, kernel_size=2)
F.adaptive_avg_pool2d(x, (1, 1))

# 归一化
F.normalize(x, p=2, dim=-1)  # L2 归一化
F.layer_norm(x, normalized_shape=[4, 5])
F.batch_norm(x, running_mean, running_var, weight, bias)

# Dropout
F.dropout(x, p=0.5, training=True)

# 其他
F.pad(x, pad=(1, 1, 2, 2))  # 填充
F.interpolate(x, size=(8, 10), mode='bilinear')  # 插值
F.one_hot(torch.tensor([0, 1, 2]), num_classes=5)  # one-hot 编码
```

## 16. 常用技巧

```python
# 设置随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Attention Mask
seq_len = 5
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # 上三角 mask

# Padding Mask
lengths = torch.tensor([3, 5, 2])
max_len = 5
mask = torch.arange(max_len)[None, :] < lengths[:, None]  # [batch, seq_len]

# 梯度累积
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 数据并行
model = nn.DataParallel(model)  # 单机多卡
model = nn.parallel.DistributedDataParallel(model)  # 分布式

# EMA（指数移动平均）
from torch.optim.swa_utils import AveragedModel
ema_model = AveragedModel(model)
```

## 17. 广播机制

```python
# 广播规则：从右往左比较维度，维度相等或其中一个为1则可以广播
x = torch.randn(3, 1, 5)
y = torch.randn(1, 4, 5)
z = x + y  # 结果: [3, 4, 5]

# 手动广播
x.expand(3, 4, 5)  # 不复制数据
x.expand_as(y)  # 扩展到与 y 相同形状
```

## 18. 常用组合操作

```python
# Softmax + Log
F.log_softmax(logits, dim=-1)  # 比 log(softmax(x)) 更稳定

# Attention Score
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
attn = F.softmax(scores, dim=-1)
output = torch.matmul(attn, V)

# Layer Norm
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + 1e-5)

# Positional Encoding
position = torch.arange(seq_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
pe = torch.zeros(seq_len, d_model)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

## 19. 数据加载

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 使用
for batch_x, batch_y in loader:
    pass
```

## 20. 常用函数速查

```python
# 张量信息
x.size()  # 形状
x.shape  # 形状
x.numel()  # 元素总数
x.dim()  # 维度数

# 修改形状
x.view()  # 重塑
x.reshape()  # 重塑（更灵活）
x.transpose()  # 转置
x.permute()  # 重排维度
x.squeeze()  # 删除维度1
x.unsqueeze()  # 增加维度

# 拼接
torch.cat()  # 在已有维度拼接
torch.stack()  # 在新维度拼接
torch.chunk()  # 分割
torch.split()  # 分割

# 数学
torch.sum()
torch.mean()
torch.max()
torch.min()
torch.matmul()  # 矩阵乘法
torch.bmm()  # batch 矩阵乘法

# 激活
F.relu()
F.gelu()
F.softmax()
F.log_softmax()

# 损失
F.cross_entropy()
F.mse_loss()
F.binary_cross_entropy_with_logits()
```

