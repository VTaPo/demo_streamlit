import bs4
import os
import asyncio
import streamlit as st
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

# Initialize Streamlit app
st.title("LangChain Q&A Generator")
st.write("Click 'Generate Responses' to get answers to predefined questions.")

# Define asynchronous function for retrieval and generation
async def generate_responses():
    # Load Documents
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Prompt and LLM
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Post-processing function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Questions
    qs = [
        'What is Task Decomposition?', 
        'What is the importance of Task Decomposition?', 
        'History of Task Decomposition'
    ]

    # Execute the chain asynchronously
    results = await rag_chain.abatch(qs)
    return results

# Streamlit button to trigger generation
if st.button("Generate Responses"):
    with st.spinner("Generating responses..."):
        responses = asyncio.run(generate_responses())
        for question, answer in zip(['What is Task Decomposition?', 'What is the importance of Task Decomposition?', 'History of Task Decomposition'], responses):
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")
            st.write("---")
