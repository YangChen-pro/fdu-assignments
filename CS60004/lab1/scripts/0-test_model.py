from openai import OpenAI
# Configured by environment variables
client = OpenAI(
    api_key="EMPTY",  # vLLM 默认不需要 API 密钥
    base_url="http://localhost:12345/v1"
)

messages = [
    {"role": "system", "content": "你是一个精通物理的老师。"},
    {"role": "user", "content": "请用一句话解释什么是量子力学。"}
]

chat_response = client.chat.completions.create(
    model="qwen-3.5-9b",
    messages=messages,
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
    }, 
)
print("Chat response:", chat_response)

print("\n\n模型输出: \n", chat_response.choices[0].message.content)
