
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import os
from dotenv import load_dotenv
load_dotenv()


class ResponderAgent:
    def __init__(self):
        # 从环境变量加载 system1_model 的模型名称和 base_url
        self.system1_model = ChatOllama(
            model=os.getenv("OLLAMA_RESPONDER_LLM_1", "llama3.1"),
            base_url=os.getenv("OLLAMA_RESPONDER_LLM_1_BASE_URL", "http://localhost:11434"),
            temperature=0
        )
        # 从环境变量加载 system2_model 的模型名称和 base_url
        self.system2_model = ChatOllama(
            model=os.getenv("OLLAMA_RESPONDER_LLM_2", "deepseek-r1:8b"),
            base_url=os.getenv("OLLAMA_RESPONDER_LLM_2_BASE_URL", "http://localhost:11434"),
            temperature=0
        )
        self.responder_prompt_template = """
# Task
You are an intelligent responder agent. Given a user question, reply in a fun witty manner. 
Reply in whatever language the question is in.

Question: {question}
"""

        self.responder_wRAG_prompt_template = """
# Task
You are an intelligent responder agent. Given a user question, and retrieved context, strictly follow the information given the retrieved context.
If the information is not found, just say it is not found.
Do not make up your own answer 
Reply in whatever language the question is in a fun witty manner.

Retrieved Context: {retrieved_info}
Question: {question}
"""

    def run(self, question, context="", req_think=False):
        if context == "":
            responder_prompt = ChatPromptTemplate.from_template(
                self.responder_prompt_template
            )
        else:
            responder_prompt = ChatPromptTemplate.from_template(
                self.responder_wRAG_prompt_template
            )
        if not req_think:
            llm = self.system1_model
        else:
            llm = self.system2_model

        responder_chain = responder_prompt | llm
        if context == "":
            result = responder_chain.invoke({"question": question})
        else:
            result = responder_chain.invoke(
                {"question": question, "retrieved_info": context}
            )
        return result.content

    def invoke(self, state):
        question = state.get("question")
        if state.get("retrieved_info"):
            context = state.get("retrieved_info")
        else:
            context = ""
        responder_reply = self.run(
            question, context, state.get("router_need_system_2")
        )
        state["responder_reply"] = responder_reply
        return state

if __name__ == "__main__":
    question = "how 'r's are in strawberry?"
    responder_agent = ResponderAgent()
    result = responder_agent.run(question, req_think=True)
    print(result)