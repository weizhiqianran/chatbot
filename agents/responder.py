from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI  # 添加 OpenAI 支持

import os
from dotenv import load_dotenv
load_dotenv()


class ResponderAgent:
    def __init__(self):
        # System 1 模型配置
        model_type_1 = os.getenv("RESPONDER_MODEL_TYPE_1", "ollama").lower()
        if model_type_1 == "ollama":
            self.system1_model = ChatOllama(
                model=os.getenv("OLLAMA_RESPONDER_LLM_1", "llama3.1"),
                base_url=os.getenv("OLLAMA_RESPONDER_LLM_1_BASE_URL", "http://localhost:11434"),
                temperature=0
            )
        elif model_type_1 == "openai":
            self.system1_model = ChatOpenAI(
                model=os.getenv("OPENAI_RESPONDER_LLM_1", "gpt-3.5-turbo"),
                api_key=os.getenv("OPENAI_RESPONDER_API_KEY_1"),
                base_url=os.getenv("OPENAI_RESPONDER_BASE_URL_1", None),
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported model type for system1: {model_type_1}")

        # System 2 模型配置
        model_type_2 = os.getenv("RESPONDER_MODEL_TYPE_2", "ollama").lower()
        if model_type_2 == "ollama":
            self.system2_model = ChatOllama(
                model=os.getenv("OLLAMA_RESPONDER_LLM_2", "deepseek-r1:8b"),
                base_url=os.getenv("OLLAMA_RESPONDER_LLM_2_BASE_URL", "http://localhost:11434"),
                temperature=0
            )
        elif model_type_2 == "openai":
            self.system2_model = ChatOpenAI(
                model=os.getenv("OPENAI_RESPONDER_LLM_2", "gpt-4"),
                api_key=os.getenv("OPENAI_RESPONDER_API_KEY_2"),
                base_url=os.getenv("OPENAI_RESPONDER_BASE_URL_2", None),
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported model type for system2: {model_type_2}")

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