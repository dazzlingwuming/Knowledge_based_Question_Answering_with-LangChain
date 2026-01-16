import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# LangChain 1.0 新版依赖
from langchain_openai import ChatOpenAI ,OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
os.environ["OPENAI_API_BASE"]  = os.getenv("ZHIPU_API_URL")

app = FastAPI()

# 配置跨域
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 初始化LLM
llm = OpenAI(model="Qwen/Qwen2.5-Coder-32B-Instruct", temperature=0.7)

# 2. 定义新版Prompt模板（更简洁，适配1.0）
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个精准、简洁的聊天助手，严格遵守以下规则：
1. 仅回答用户当前提出的问题，绝不模拟、虚构任何后续的用户提问，也不模拟自己的再次回复；
2. 回复仅保留针对本次问题的核心答案，简洁明了，无多余内容；
3. 即使用户的问题涉及对话格式，也仅解释问题本身，不生成示例对话；
4. 回复中禁止出现「用户：」「AI：」「提问：」「回答：」等对话格式标识。"""),
    MessagesPlaceholder(variable_name="history"),  # 对话历史
    ("human", "{input}")  # 用户输入（纯字符串）
])

# 3. 构建基础链
base_chain = prompt | llm

# 4. 会话历史存储（按session_id区分）
session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# 5. 包装带历史的链
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# 请求体模型
class QAQuery(BaseModel):
    question: str
    session_id: str = "default"


# 问答接口
@app.post("/api/qa")
async def qa_endpoint(query: QAQuery):
    try:
        question = query.question.strip()
        if not question:
            return {"code": 400, "msg": "问题不能为空", "data": ""}

        # 1. 从请求中获取前端传递的session_id
        current_session_id = query.session_id

        # 2. 将session_id传入config的configurable参数中
        response = chain_with_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": current_session_id}}  # 关键修复：传入正确的session_id
        )

        answer = response.strip() if response.strip() else "抱歉，我暂时无法回答这个问题。"
        return {"code": 200, "msg": "success", "data": answer}
    except Exception as e:
        return {"code": 500, "msg": f"服务异常：{str(e)}", "data": ""}