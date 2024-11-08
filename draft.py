import bs4
import os
import asyncio
import streamlit as st
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

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

    # Prompt and LLM
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question: {question}"
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Post-processing function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"question": RunnablePassthrough()}
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
