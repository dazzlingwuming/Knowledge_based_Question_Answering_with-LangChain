#为了防止模型模拟多轮对话，避免模糊表述









#==================提示词直接写入模型调用脚本===================================================
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from data.conversa_history.历史保存 import load_memory_from_file, save_memory_to_file
from llm_base_model import init_llm_with_memory

system_template = """你是一个精准、简洁的聊天助手，严格遵守以下规则：
1. 仅回答用户当前提出的问题，**绝不模拟、虚构任何后续的用户提问，也不模拟自己的再次回复**；
2. 回复仅保留针对本次问题的核心答案，简洁明了，无多余内容；
3. 即使用户的问题涉及对话格式，也仅解释问题本身，不生成示例对话；
4. 回复中禁止出现「用户：」「AI：」「提问：」「回答：」等对话格式标识。"""


def prompt_creat(message: str) -> str:
    """创建格式化的提示语，包含历史对话示例"""
    human_template = "{question}"
    history_chat = "{history}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        # ("history", history_chat),  # 历史问题在历史保存脚本中处理
        ("human", human_template)  # 当前问题
    ])

    # 格式化当前问题
    formatted_chat_prompt = chat_prompt.format_prompt(question=message)

    return formatted_chat_prompt


if __name__ == "__main__":
    message = "我是随？你喜欢我吗？"
    format_prompt = prompt_creat(message)
    print(type(format_prompt))
    print(format_prompt)
    #结合历史保存来看效果
    chat_chain = init_llm_with_memory()
    load_memory_from_file(chat_chain.memory,session_id="小明")  # 启动时加载
    response = chat_chain.invoke(format_prompt)
    # 对话结束后保存
    save_memory_to_file(chat_chain.memory,session_id="小明")
    print(response)