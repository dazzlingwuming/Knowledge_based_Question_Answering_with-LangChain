import json
import os
from datetime import datetime

# from llm_base_model import init_llm_with_memory
# from promt提示词 import prompt_creat


# 新增：记忆持久化功能（保存到JSON文件）
def save_memory_to_file(memory, session_id="default_user", file_path="conversation_history.json"):
    """保存记忆到JSON文件"""
    # 获取历史对话
    history = memory.chat_memory.messages
    # 转换为可序列化格式
    history_dict = [
        {
            "type": msg.type,
            "content": msg.content,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        } for msg in history
    ]
    # 按用户ID存储（支持多用户）
    data = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[session_id] = history_dict
    # 写入文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_memory_from_file(memory, session_id="default_user", file_path="conversation_history.json"):
    """从JSON文件加载记忆"""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if session_id not in data:
        return
    # 加载历史对话到记忆
    for msg in data[session_id]:
        if msg["type"] == "human":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            memory.chat_memory.add_ai_message(msg["content"])


def clear_memory(memory):
    """清空当前记忆"""
    memory.chat_memory.clear()
    print("✅ 历史对话已清空！")


# 调用示例（在初始化对话链后）
if __name__ == "__main__":
    # chat_chain = init_llm_with_memory()
    # load_memory_from_file(chat_chain.memory,session_id="小明")  # 启动时加载
    # # message = "我叫小明，今年25岁，请你写一首关于春天的的诗歌。"
    # # format_prompt = prompt_creat(message)
    # response = chat_chain.invoke({"input": "你知道我是谁吗？"})
    # # 对话结束后保存
    # save_memory_to_file(chat_chain.memory,session_id="小明")
    # print(response["response"])
    pass