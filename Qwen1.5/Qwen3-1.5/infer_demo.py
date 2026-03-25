from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ################## 唯一修改：这里填你的Qwen1.5模型路径 ##################
model_path = r"E:\CLASS\big modle\qwen\Qwen1.5-0.5B-Chat"

# 强制本地加载，兼容Qwen1.5，绝不联网报错
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # CPU专属适配，解决兼容性问题
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True  # Qwen1.5必需参数！
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)

# 对话prompt
prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型回复：\n", response)