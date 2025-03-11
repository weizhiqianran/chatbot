# 关于
这是伴随 Medium 文章的代码仓库。

<img src="images/chat_screenshot.png" width=560px>

# 设置
Python 环境
```bash
conda env create -f env.yml
conda activate chatbot
```
创建向量存储
```bash
python create_vs.py
```
运行 Web 应用
```bash
streamlit run app.py
```

# 你可以提问的问题
- 触发 RAG（检索增强生成）：提出与 Kredivo 相关的问题
- 触发系统 2：让它规划一个行程
