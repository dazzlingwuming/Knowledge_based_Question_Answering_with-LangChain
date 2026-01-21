from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import OpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate  # 新增：导入PromptTemplate

from data.conversa_history.历史保存 import load_memory_from_file, save_memory_to_file

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

#创建可以调用工具的基础模型
def init_llm_tool_model(tools=[]):

    # 1. 初始化LLM
    llm = OpenAI(
        api_key=api_key,
        base_url=api_url,
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.3,
        max_tokens=2048,
        top_p=0.3,
    )
    # 步骤2：加载LangChain现成的维基百科Tool（无需自己写API调用）
    # WikipediaAPIWrapper：封装了维基百科API的所有底层逻辑
    # WikipediaQueryRun：把API包装成LangChain Agent可调用的Tool
    wikipedia_api = WikipediaAPIWrapper(lang="zh", top_k_results=2)  # 中文+返回3条结果
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api,description="一个可以在维基百科中搜索信息的工具。如果你想要了解某个人物、事件、地点等相关信息时，可以使用这个工具来获取准确的百科内容。", return_direct=True)

    # 步骤3：定义Agent要使用的工具列表（可添加多个，比如计算器、搜索等）
    tools = [wikipedia_tool]+tools
    #初始化记忆组件
    # 初始化记忆组件
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        ai_prefix="ai",
        human_prefix="human",
        input_key="input"
    )

    # 步骤4：初始化Agent（核心：让模型自主决定是否调用工具）
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # 适合对话+工具调用的Agent类型
        verbose=True,  # 开启调试，能看到Agent的思考过程（是否调用工具、调用哪个）
        handle_parsing_errors="返回无法解析的结果，请简化回答",  # 异常兜底
        memory=memory,  # 添加记忆组件
    )

    return agent


class LLMBaseModel:
    def __init__(self , model_type: str = "memory"):
        if model_type == "memory":
            self.llm = init_llm_with_memory()
        elif model_type == "tool":
            self.llm = init_llm_tool_model()

    def chat_completion(self, input: str) -> str:
        response = self.llm.invoke({"input": {input}})
        return response


if __name__ == "__main__":
    # chat_chain = init_llm_with_memory()
    # print("===== 带记忆的对话系统启动 =====")
    #
    # res1 = chat_chain.invoke({"input": "我是小李，请你写一首关于学习的五言绝句诗。只需要你回复诗的内容，不要其他多余的文字。"})
    # print(f"助手回答：{res1['response']}\n")
    #
    # res2 = chat_chain.invoke({"input": "我刚才说我叫什么名字？要你做了什么事情？"})
    # print(f"助手回答：{res2['response']}\n")
    # @tool
    # def abc_computer(a: str) -> int:
    #     """abc计算。"""
    #     a = int(a)
    #     return a + 5
    #
    # file_path = "data/conversa_history/conversation_history.json"
    # llm_tool = init_llm_tool_model(tools=[abc_computer])
    # #添加记忆
    # load_memory_from_file(llm_tool.memory,session_id="小明" , file_path=file_path)  # 启动时加载
    # chat_history = llm_tool.memory.chat_memory.messages  # 从加载的记忆中获取历史
    # response = llm_tool.invoke({"input": "现在我需要你将13进行abc计算，使用工具得到结果，无论结果对还是需将原始结果发送给我", "chat_history": chat_history})
    # # 对话结束后保存
    # save_memory_to_file(llm_tool.memory,session_id="小明",file_path=file_path)
    # print(response)
    pass
