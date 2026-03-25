# ① 加载工具包，指定推理设备（GPU/CPU）
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 关键：修改为你本地的Qwen2.5-0.5B-Instruct模型路径
model_path = r'E:\CLASS\big_modle\qwen\Qwen2.5-0.5B-Instruct'  

# ② 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ③ 定义对话提示语（可自行修改user的问题）
prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# ④ 格式化提示语并转换为模型可识别的张量
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# ⑤ 模型生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512  # 限制生成的最大字符数，可调整
)
# 截取仅模型生成的内容（剔除输入的提示语）
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# ⑥ 解码生成结果，输出最终回答
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)