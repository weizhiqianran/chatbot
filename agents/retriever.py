from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 添加 OpenAI 支持

import os
from dotenv import load_dotenv
load_dotenv()


class RetrieverAgent:
    def __init__(self):
        # 从环境变量读取模型类型
        model_type = os.getenv("RETRIEVER_MODEL_TYPE", "ollama").lower()

        # 初始化 LLM
        if model_type == "ollama":
            self.llm = ChatOllama(
                model=os.getenv("OLLAMA_RETRIEVER_LLM", "llama3.1"),
                base_url=os.getenv("OLLAMA_RETRIEVER_LLM_BASE_URL", "http://localhost:11434"),
                temperature=0
            )
            embedding_function = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
                base_url=os.getenv("OLLAMA_RETRIEVER_LLM_BASE_URL", "http://localhost:11434")
            )
        elif model_type == "openai":
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_RETRIEVER_LLM", "gpt-4"),
                api_key=os.getenv("OPENAI_RETRIEVER_API_KEY"),
                base_url=os.getenv("OPENAI_RETRIEVER_BASE_URL", None),  # 可选 vLLM 端点
                temperature=0
            )
            embedding_function = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                api_key=os.getenv("OPENAI_RETRIEVER_API_KEY"),
                base_url=os.getenv("OPENAI_RETRIEVER_BASE_URL", None)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'ollama' or 'openai'.")

        self.search_term_prompt_template = """
# Task
You are an intelligent search term suggestion agent. Given a user question, suggest search english terms, up to 3 words, which will optimize the vector search.
Strictly Return your response with just the search term.

Question: {question}
"""
        self.search_term_prompt = ChatPromptTemplate.from_template(
            self.search_term_prompt_template
        )
        self.search_term_chain = self.search_term_prompt | self.llm
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3_000,
            chunk_overlap=50,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=25,
        )

        self.fs = LocalFileStore(f"knowledge_base/local_file_store")
        self.store = create_kv_docstore(self.fs)

        self.vectorstore = Chroma(
            collection_name="parent_docs",
            embedding_function=embedding_function,
            persist_directory=f"knowledge_base/vectorstore",
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_kwargs={"k": 3},
        )

    def run_search_term(self, question):
        result = self.search_term_chain.invoke({"question": question})
        search_terms = result.content
        search_text = f"{search_terms}, {question}"
        return search_text

    def invoke(self, state):
        question = state.get("question")
        search_text = self.run_search_term(question)
        relevant_docs = self.retriever.invoke(search_text)
        retrieved_info = ""
        for d in relevant_docs:
            retrieved_info += d.page_content
            retrieved_info += "\n"
        state["search_terms"] = search_text
        state["retrieved_info"] = retrieved_info

        return state

if __name__ == "__main__":
    question = "who founded kredivo?"
    retriever_agent = RetrieverAgent()
    retriever_agent.invoke({"question": question})