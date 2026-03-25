import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 你的本地模型路径
model_path = r"E:\CLASS\big_modle\qwen\Qwen3.0-1.7B"

# 加载模型（强制离线，无报错）
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

# 对话核心函数
def chat(message, history):
    messages = []
    # 拼接历史对话
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    # Qwen3 对话模板
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response

# 启动界面（删除了不兼容的theme参数）
demo = gr.ChatInterface(
    fn=chat,
    title="Qwen3 本地聊天助手",
    description="基于通义千问3.0大模型 | 本地离线部署"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8000)