import streamlit as st
from graph import workflow

st.title("Starter Pack Chatbot")
st.markdown("未冉之芊")

def stream_data():
    for msg, metadata in workflow.stream(
        {
            "question": prompt,
        },
        stream_mode="messages",
    ):
        if metadata["langgraph_node"] == "responder":
            stream = msg.content
            yield stream

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        print(prompt)
        response = st.write_stream(stream_data)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 自动刷新页面
    st.rerun()
