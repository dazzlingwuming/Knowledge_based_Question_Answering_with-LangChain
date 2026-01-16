from fast端口 import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import asyncio

# 初始化FastAPI应用
app = FastAPI()

# 1. 提前初始化LLM和问答链（避免每次请求重复初始化）
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
prompt = PromptTemplate(
    input_variables=["question"],
    template="请清晰、准确地回答以下问题：{question}"
)
qa_chain = LLMChain(llm=llm, prompt=prompt)

# 定义请求体格式
class QAQuery(BaseModel):
    question: str

# 2. 定义问答接口
@app.post("/api/qa")
async def qa_endpoint(query: QAQuery):
    try:
        if not query.question.strip():
            return {"code": 400, "msg": "问题不能为空", "data": ""}
        # 调用问答链（LLMChain.run 是同步/阻塞的；在 async 端点中放到线程池执行）
        answer = await asyncio.to_thread(qa_chain.run, question=query.question)
        return {"code": 200, "msg": "success", "data": answer}
    except Exception as e:
        return {"code": 500, "msg": f"服务异常：{str(e)}", "data": ""}

# 启动命令：uvicorn app:app --reload

