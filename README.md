# Qwen 系列大语言模型（1.5/2.5/3.0-0.5B）部署与使用说明

本文档主要介绍阿里通义千问 Qwen1.5、Qwen2.5、Qwen3.0 轻量版（0.5B，5 亿参数）的本地部署、模型推理及 ChatBot 网页应用构建方法，该系列轻量模型可在普通个人电脑运行，适合大语言模型入门学习，涵盖模型加载、推理、Web 交互界面搭建全流程。

## 一、项目简介

本项目基于阿里达摩院 Qwen 系列开源大语言模型，选择0.5B 参数量的轻量版本，适配普通 PC 的 CPU/GPU 环境，无高端算力要求。通过本项目可掌握：

1. 开源大语言模型的权重加载、Tokenizer 配置与推理方法；
2. 基于 FastAPI+Gradio 构建大语言模型 ChatBot 网页应用；
3. 大语言模型 Web 交互应用的核心构建逻辑。

支持模型版本：

- Qwen1.5-0.5B-Chat
- Qwen2.5-0.5B-Instruct
- Qwen3.0-1.7B

## 二、环境准备

### 2.1 基础环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.12（推荐 2.0+，CPU/GPU 版本均可）
- CUDA ≥ 11.4（仅 GPU 环境需要，CPU 环境可忽略）

### 2.2 虚拟环境创建（推荐，避免包冲突）

使用 conda 创建独立虚拟环境，命名为`Qwen`（可自定义）：

```
conda create -n Qwen python=3.8
conda activate Qwen
```

### 2.3 依赖包安装

依次执行以下命令安装所有项目依赖，**Qwen1.5/2.5/3.0 版本依赖完全一致**：

```
# 安装PyTorch（CPU版本，GPU版本请参考PyTorch官网替换命令）
pip install torch torchvision torchaudio
# 安装大模型核心依赖
pip install transformers>=4.32 accelerate>=0.26.0
# 安装模型下载工具
pip install modelscope
# 安装ChatBot网页应用依赖
pip install gradio fastapi
```

## 三、源码与模型权重下载

### 3.1 项目源码下载

分别克隆 Qwen1.5/2.5/3.0 官方源码仓库，建议在本地新建统一目录执行克隆：

```
# 克隆Qwen1.5源码
git clone https://github.com/QwenLM/Qwen1.5.git
# 克隆Qwen2.5源码
git clone https://github.com/QwenLM/Qwen2.5.git
# 克隆Qwen3.0源码
git clone https://github.com/QwenLM/Qwen3.0.git
```

> 若 git 克隆失败，可前往对应 GitHub 仓库手动下载源码压缩包并解压。

### 3.2 模型权重下载

模型权重推荐通过**ModelScope 魔搭社区**下载（自动适配环境，避免手动下载文件缺失），也可手动前往 ModelScope 官网下载。**所有操作均在上述创建的`Qwen`虚拟环境中执行**。

#### 方法 1：推荐（ModelScope 代码下载，指定本地目录，避免 C 盘缓存）

新建 Python 脚本（如`download_model.py`），根据需要选择对应模型版本的代码，执行脚本即可完成权重下载，**cache_dir 为自定义模型保存目录**（如`./model_weights/`）：

```
from modelscope import snapshot_download

# 选择对应模型版本，取消注释即可
## Qwen1.5-0.5B-Chat
# model_dir = snapshot_download('qwen/Qwen1.5-0.5B-Chat', cache_dir='./model_weights/')
## Qwen2.5-0.5B-Instruct
# model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./model_weights/')
## Qwen3.0-0.5B-Instruct
# model_dir = snapshot_download('qwen/Qwen3.0-0.5B-Instruct', cache_dir='./model_weights/')

print(f"模型权重下载完成，路径：{model_dir}")
```

```
python download_model.py
```

#### 方法 2：手动下载（不推荐）

前往 ModelScope 对应模型地址，下载所有文件并保存至本地指定目录：

- Qwen1.5-0.5B-Chat：https://www.modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/files
- Qwen2.5-0.5B-Instruct：https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files
- Qwen3.0-0.5B-Instruct：https://modelscope.cn/models/Qwen/Qwen3-1.7B/files

## 四、模型加载与本地推理

Qwen1.5/2.5/3.0 的加载与推理代码**架构完全一致**，仅需修改模型路径`model_path`为本地实际的模型权重保存路径，以下为通用推理代码，支持 CPU/GPU 自动适配。

### 4.1 通用推理代码

新建 Python 脚本（如`model_infer.py`），复制以下代码，**修改`model_path`为步骤 3.2 中模型的实际保存路径**：

```
# 1. 加载工具包
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 2. 配置设备与模型路径（核心：修改为本地实际模型路径）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 示例：根据模型版本替换
# model_path = './model_weights/qwen/Qwen1.5-0.5B-Chat'  # Qwen1.5
# model_path = './model_weights/qwen/Qwen2.5-0.5B-Instruct'  # Qwen2.5
model_path = './model_weights/qwen/Qwen3.0-0.5B-Instruct'  # Qwen3.0

# 3. 加载模型与Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 4. 定义对话提示语
prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 5. 格式化提示语并转换为模型输入张量
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 6. 模型推理生成内容
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512  # 最大生成token数，可自定义
)
# 截取生成的内容（排除输入部分）
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 7. 解码生成结果并打印
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("模型回复：", response)
```

### 4.2 执行推理

在虚拟环境中运行脚本，即可得到模型推理结果：

```
python model_infer.py
```

![17b47be85a544c239988e86d1c6e10f5](C:\Users\毛姐\AppData\Local\Temp\17b47be85a544c239988e86d1c6e10f5.png)

![3a7d02f277b0436baefd949dfd57caa9](C:\Users\毛姐\AppData\Local\Temp\3a7d02f277b0436baefd949dfd57caa9.png)

![a499e201cfde4d8f82d13b5d26b74e5c](C:\Users\毛姐\AppData\Local\Temp\a499e201cfde4d8f82d13b5d26b74e5c.png)

## 五、构建 ChatBot 网页交互应用

基于 Gradio+FastAPI 构建网页版 ChatBot，**三个模型版本的构建流程完全一致**，仅需修改源码中模型权重的路径配置，运行后可通过浏览器访问本地地址实现对话交互。

### 5.1 路径配置修改

1. 进入对应模型的源码目录，打开 Web 应用脚本：
   - Qwen1.5：`./Qwen1.5/examples/demo/web_demo.py`
   - Qwen2.5：`./Qwen2.5/examples/demo/web_demo.py`
   - Qwen3.0：`./Qwen3.0/examples/demo/web_demo.py`
2. 找到脚本中**第 15 行**的`DEFAULT_CKPT_PATH`配置项，将其值修改为**本地模型权重的实际路径**（绝对路径 / 相对路径均可）。

**配置示例**：

```
# 替换前
# Qwen1.5: DEFAULT_CKPT_PATH = 'Qwen/Qwen1.5-7B-Chat'
# Qwen2.5: DEFAULT_CKPT_PATH = 'Qwen/Qwen2.5-7B-Instruct'
# Qwen3.0: DEFAULT_CKPT_PATH = 'Qwen/Qwen3.0-7B-Instruct'

# 替换后（以绝对路径为例，根据本地实际路径修改）
DEFAULT_CKPT_PATH = 'D:/Qwen_series/model_weights/qwen/Qwen3.0-0.5B-Instruct'
```

### 5.2 运行 ChatBot 网页应用

在虚拟环境中，进入`web_demo.py`所在的目录，执行以下命令启动应用：

```
python web_demo.py
```

### 5.3 访问与使用

启动成功后，打开浏览器，输入本地访问地址：

![e11116c523244dceafa061a3c4b68db0](C:\Users\毛姐\AppData\Local\Temp\e11116c523244dceafa061a3c4b68db0.png)

![a5e0c88c2e6d4f7a81da97d2f176dde2](C:\Users\毛姐\AppData\Local\Temp\a5e0c88c2e6d4f7a81da97d2f176dde2.png)

![7f07c6c6abb44634a13d8d159f0a49be](C:\Users\毛姐\AppData\Local\Temp\7f07c6c6abb44634a13d8d159f0a49be.png)

在网页的对话框中输入问题，点击`Submit`即可与模型对话，支持清空历史、重新生成等功能。

![60516ec56b604ac191b6b2b8761e2f0b](C:\Users\毛姐\AppData\Local\Temp\60516ec56b604ac191b6b2b8761e2f0b.png)

![ca95b8a3f16f47068784cb14e2a4dc0e](C:\Users\毛姐\AppData\Local\Temp\ca95b8a3f16f47068784cb14e2a4dc0e.png)

![2fde18c9657f42808d85745b931933c2](C:\Users\毛姐\AppData\Local\Temp\2fde18c9657f42808d85745b931933c2.png)

## 六、ChatBot 核心构建逻辑

Qwen 系列 ChatBot 网页应用由**三层架构**组成，三个版本的核心逻辑完全一致：

1. **模型权重服务**：加载本地模型权重与 Tokenizer，提供推理能力；
2. **API 服务**：基于 FastAPI 将模型推理过程封装为可调用的 API 接口，供外部程序调用；
3. **Web 服务**：基于 Gradio 构建用户友好的网页界面，无需前端开发知识，实现提示词输入、结果展示、交互按钮等功能。

**用户交互流程**：

浏览器访问 Web 页面 → 输入提示词并提交 → 前端将请求发送至 FastAPI 接口 → 接口调用本地模型进行推理 → 推理结果返回至后端 → 后端将结果展示在前端网页。







