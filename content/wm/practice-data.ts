// 第一幕（01-04 课）的配套练习：锚定 ctallec/world-models 仓库。
// 每课给出：跑什么命令、证明什么、不证明什么。命令与仓库 README 一致；
// 旧依赖与 gymnasium CarRacing-v3 的差异，按各课正文的补丁说明处理。

export interface PracticeStep {
  title: string;
  detail: string;
  command: string;
  expected: string;
}

export interface LessonPractice {
  lessonId: string;
  purpose: string;
  proves: readonly string[];
  doesNotProve: string;
  steps: readonly PracticeStep[];
  repositoryName?: string;
  repositoryUrl?: string;
  includeSetup?: boolean;
}

export const practiceRepositoryName = "ctallec/world-models";
export const practiceRepositoryUrl = "https://github.com/ctallec/world-models";

// 首次使用的公共准备步骤：克隆仓库、建独立环境、装依赖。只需完成一次。
export const practiceSetupSteps: readonly PracticeStep[] = [
  {
    title: "下载锚定仓库",
    detail: "在准备存放项目的目录中执行。",
    command: "git clone https://github.com/ctallec/world-models.git",
    expected: "当前目录出现 world-models 文件夹。",
  },
  {
    title: "进入仓库目录",
    detail: "后续所有命令都在仓库根目录执行。",
    command: "cd world-models",
    expected: "终端当前目录以 /world-models 结尾。",
  },
  {
    title: "确认 Python 版本",
    detail: "建议 Python 3.10 或更高；版本太老装不上新版 gymnasium。",
    command: "python3 --version",
    expected: "输出 Python 3.10 或更高版本号。",
  },
  {
    title: "创建独立环境",
    detail: "只建本课程用的环境，不动系统 Python。",
    command: "python3 -m venv .venv",
    expected: "当前目录出现 .venv 文件夹，命令没有报错。",
  },
  {
    title: "启用独立环境",
    detail: "启用后终端提示符通常出现 (.venv)。",
    command: "source .venv/bin/activate",
    expected: "命令没有报错，后续 pip 安装进这个环境。",
  },
  {
    title: "安装依赖",
    detail:
      "仓库的依赖清单写于 2018 年代。装不上时不要硬装，按第 01 课正文的补丁说明改用 gymnasium 方案。",
    command: "pip3 install -r requirements.txt",
    expected: "依赖安装完成；个别包装不上时，第 01 课正文有替代清单。",
  },
];

const practices = [
  {
    lessonId: "01",
    purpose:
      "把环境装通，用随机策略采出第一份 CarRacing 轨迹数据集，看清观察、动作、奖励的原始形状。",
    proves: [
      "环境和渲染能跑，数据管线能把轨迹落盘。",
      "你手里有了第 02-04 课都要用的第一份数据。",
    ],
    doesNotProve:
      "这一步不训练任何模型，也不说明随机策略的数据足够训出好的世界模型；数据够不够，第 04 课回真实环境验收时见分晓。",
    steps: [
      {
        title: "采集随机策略轨迹",
        detail:
          "第一次跑建议把 --rollouts 降到 50、--threads 降到本机核数，先验证流程再放大。",
        command:
          "python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8",
        expected:
          "datasets/carracing 下出现按线程分目录的 rollout 文件，每个文件存一条轨迹的观察、动作序列。",
      },
    ],
  },
  {
    lessonId: "02",
    purpose:
      "训练 V（VAE），把 96×96 的画面压成 32 维 latent，并检查重建质量。",
    proves: [
      "VAE 能在自己采的数据上收敛，重建图像能看出路面结构。",
      "checkpoint 已落盘，第 03 课可以直接接着用。",
    ],
    doesNotProve:
      "重建好看不代表 latent 保住了所有决策相关的信息；哪些信息被扔了、要不要紧，课内的插值走查和 β 对比实验才能回答。",
    steps: [
      {
        title: "训练 VAE",
        detail:
          "--logdir 指向实验目录，中断后重跑会自动续训；想从头开始加 --noreload。",
        command: "python trainvae.py --logdir exp_dir",
        expected:
          "exp_dir/vae 下出现 checkpoint，重建 loss 随 epoch 下降，采样重建图逐渐能认出赛道。",
      },
    ],
  },
] satisfies readonly LessonPractice[];

const practicesLate = [
  {
    lessonId: "03",
    purpose:
      "训练 M（MDN-RNN），学“当前 latent + 动作”到“下一个 latent 的分布”的映射。",
    proves: [
      "动作条件的动力学模型能训起来，负对数似然随训练下降。",
      "模型输出的是分布参数（多组均值方差），不是单点。",
    ],
    doesNotProve:
      "损失下降不自动证明模型把动作用起来了；动作对换对照实验在课内做，分不了岔就得回来查。",
    steps: [
      {
        title: "训练 MDN-RNN",
        detail:
          "必须先在同一个 exp_dir 里训练过 VAE（第 02 课），这个脚本要读它的 checkpoint。",
        command: "python trainmdrnn.py --logdir exp_dir",
        expected:
          "exp_dir/mdrnn 下出现 checkpoint，训练与验证损失逐 epoch 下降。",
      },
    ],
  },
  {
    lessonId: "04",
    purpose:
      "用 CMA-ES 训练 867 个参数的线性控制器，再评测整条 V-M-C 链路能不能开车。",
    proves: [
      "V、M、C 三个零件能合成一个真的会开车的智能体。",
      "真实环境分数与论文方向一致（复现看方向，不逐点对齐原文数字）。",
    ],
    doesNotProve:
      "这两条命令只覆盖真实环境训练路线；梦境训练与温度 τ 实验需要课内给的胶水代码，不在仓库自带脚本里。",
    steps: [
      {
        title: "训练控制器",
        detail:
          "多进程评估吃 CPU 核数；服务器上没有显示器时，整条命令包进 xvfb-run（写法见仓库 README）。",
        command:
          "python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display",
        expected:
          "exp_dir/ctrl 下持续更新 best 参数，日志里 best return 逐代上升。",
      },
      {
        title: "评测控制器",
        detail: "载入训练出的最优参数，在真实环境里跑分。",
        command: "python test_controller.py --logdir exp_dir",
        expected: "输出若干条评测回合的回报；与论文同方向即算达标。",
      },
    ],
  },
] satisfies readonly LessonPractice[];

const laterPractices = [
  {
    lessonId: "23",
    purpose:
      "在 C-SWM 官方小环境上训练槽化动力学，确认槽和物体对得上。",
    proves: [
      "Shapes 环境能训起来，Hits@1 明显高于随机。",
      "槽可视化能看出两个物体不在同一个槽里。",
    ],
    doesNotProve:
      "这是方向性复现，不追论文 Table 1 的多种子均值。桌面真实杯子另算。",
    repositoryName: "tkipf/c-swm",
    repositoryUrl: "https://github.com/tkipf/c-swm",
    includeSetup: false,
    steps: [
      {
        title: "克隆官方仓库",
        detail: "Python 环境独立建，不要和 CarRacing 那套旧依赖混装。",
        command: "git clone https://github.com/tkipf/c-swm.git",
        expected: "当前目录出现 c-swm 文件夹。",
      },
      {
        title: "按 README 生成 Shapes 数据并训练",
        detail: "命令以仓库当前 README 为准；课内有 gym / PyTorch 版本补丁。",
        command: "python train.py --dataset data/shapes_train.h5 --encoder small --name shapes",
        expected: "出现 checkpoint；随后用 eval.py 看 1/5/10 步 Hits@1。",
      },
    ],
  },
  {
    lessonId: "25",
    purpose:
      "用 LeRobot 回放公开桌面臂轨迹，并训练一个缩小的 ACT。",
    proves: [
      "能回放一条抓放轨迹。",
      "ACT 小配置能开始训练，损失下降。",
    ],
    doesNotProve:
      "离线损失下降不等于真机成功率。没有手臂也能做完本课。",
    repositoryName: "huggingface/lerobot",
    repositoryUrl: "https://github.com/huggingface/lerobot",
    includeSetup: false,
    steps: [
      {
        title: "按当前文档安装 LeRobot",
        detail: "Python 版本和 extras 以 huggingface.co/docs/lerobot/installation 为准。",
        command: "git clone https://github.com/huggingface/lerobot.git",
        expected: "仓库克隆成功；再按文档装训练依赖。",
      },
    ],
  },
  {
    lessonId: "30",
    purpose:
      "在自己的桌子或 Genesis 桌面场景上训一个会听动作的小世界模型。",
    proves: [
      "动作对换必须分岔。",
      "有一条多步漂移曲线，以及动作置零或打乱的负对照。",
    ],
    doesNotProve:
      "这不是 DINO-WM 论文数字的复现。架构只复用前面课已跑通的一种。",
    repositoryName: "gaoyuezhou/dino_wm",
    repositoryUrl: "https://github.com/gaoyuezhou/dino_wm",
    includeSetup: false,
    steps: [
      {
        title: "对照官方特征动力学仓库",
        detail: "主实验用课内胶水脚本；这个仓库只作架构对照。",
        command: "git clone https://github.com/gaoyuezhou/dino_wm.git",
        expected: "能打开 models/visual_world_model.py，对照 concat 动作的位置。",
      },
    ],
  },
] satisfies readonly LessonPractice[];

export const practiceByLessonId = Object.fromEntries(
  [...practices, ...practicesLate, ...laterPractices].map((practice) => [
    practice.lessonId,
    practice,
  ]),
) as Record<string, LessonPractice>;

