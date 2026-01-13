from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate  # 新增：导入PromptTemplate

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")
api_url = os.getenv("ZHIPU_API_URL")
model_names = "Qwen/Qwen3-8B"

# 单轮对话模型封装
def init_zhipu_llm():
    llm = OpenAI(
        api_key=api_key,
        base_url=api_url,
        model=model_names,
        temperature=1.0,
        max_tokens=2048,
        timeout=20
    )
    return llm

# 多轮对话模型封装（整合自定义Prompt模板）
def init_llm_with_memory():
    # 1. 初始化LLM
    llm = OpenAI(
        api_key=api_key,
        base_url=api_url,
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.3,
        max_tokens=2048,
        top_p=0.3,
    )

    # 2. 初始化记忆组件（memory_key必须和Prompt模板中的{history}一致）
    memory = ConversationBufferMemory(
        memory_key="history",  # 对应Prompt模板中的{history}变量
        return_messages=True,
        ai_prefix="Assistant",
        human_prefix="User"
    )

    # 3. 自定义Prompt模板（包含你的系统规则+history+input变量）
    CUSTOM_PROMPT_TEMPLATE = """你是一个精准、简洁的聊天助手，严格遵守以下规则：
1. 仅回答用户当前提出的问题，**绝不模拟、虚构任何后续的用户提问，也不模拟自己的再次回复**；
2. 回复仅保留针对本次问题的核心答案，简洁明了，无多余内容；
3. 即使用户的问题涉及对话格式，也仅解释问题本身，不生成示例对话；
4. 回复中禁止出现「用户：」「AI：」「提问：」「回答：」等对话格式标识。

当前对话历史：
{history}

用户当前问题：
{input}

AI回复："""

    # 4. 创建PromptTemplate（绑定history和input变量）
    prompt = PromptTemplate(
        input_variables=["history", "input"],  # 声明模板中要替换的变量
        template=CUSTOM_PROMPT_TEMPLATE
    )

    # 5. 构建对话链（关联LLM、记忆、自定义Prompt）
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,  # 替换默认Prompt为自定义模板
        verbose=True    # 开启后可看到模板填充后的完整Prompt（方便调试）
    )

    return conversation_chain


class LLMBaseModel:
    def __init__(self):
        self.llm = init_llm_with_memory()

    def chat_completion(self, input: str) -> str:
        response = self.llm.invoke({"input": {input}})
        return response


if __name__ == "__main__":
    chat_chain = init_llm_with_memory()
    print("===== 带记忆的对话系统启动 =====")

    res1 = chat_chain.invoke({"input": "我是小李，请你写一首关于学习的五言绝句诗。只需要你回复诗的内容，不要其他多余的文字。"})
    print(f"助手回答：{res1['response']}\n")

    res2 = chat_chain.invoke({"input": "我刚才说我叫什么名字？要你做了什么事情？"})
    print(f"助手回答：{res2['response']}\n")