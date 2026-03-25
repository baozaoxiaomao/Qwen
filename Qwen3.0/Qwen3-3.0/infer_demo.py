from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 你本地的Qwen3模型路径（直接用）
model_path = r"E:\CLASS\big_modle\qwen\Qwen3.0-1.7B"

# 核心：新版库原生支持Qwen3，只需要这两个参数
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制离线，只读本地文件
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True  # 强制离线
)

# 测试对话
prompt = "你好，介绍一下通义千问3.0"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=256, 
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.7
)

# 输出结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))