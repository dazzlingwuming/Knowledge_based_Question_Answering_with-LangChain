from tkinter.font import names

from llm_base_model import init_zhipu_llm


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

if __name__ == "__main__":
    t1()