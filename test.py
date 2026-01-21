from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

from llm_model.llm_base_model import init_zhipu_llm


def t1():
    llm = init_zhipu_llm()
    # 测试用例列表
    test_cases = [
        "通用问题：什么是LangChain框架？",
    ]

    # 执行测试
    for case in test_cases:
        print(f"问题：{case}")
        try:
            response = llm.invoke(case)
            print(f"回答：{response}\n")
        except Exception as e:
            print(f"调用失败：{e}\n")

#测试维基百科
def t2():
    # 初始化工具 可以根据需要进行配置
    api_wrapper = WikipediaAPIWrapper(lang = "zh",top_k_results=1, doc_content_chars_max=1000)

    class WikiInputs(BaseModel):
        """维基百科工具的输入。"""
        query: str = Field(
            description="维基百科中的查询"
        )

    tool = WikipediaQueryRun(
        name="wiki-tool",
        description="在维基百科中查找内容",
        args_schema=WikiInputs,#指定输入参数的结构
        api_wrapper=api_wrapper,#传入维基百科API包装器
        return_direct=True,#表示当 Agent 调用此工具时，直接返回结果给用户，不再让 LLM 二次处理（常用于“快速回答”场景）。

    )

    # 工具默认名称
    print("name:", tool.name)
    # 工具默认的描述
    print("description:", tool.description)
    print("args:", tool.args)
    print("return_direct:", tool.return_direct)
    print(tool.run("习近平")) # 需要开启访问外网功能

if __name__ == "__main__":
    # t1()
    t2()