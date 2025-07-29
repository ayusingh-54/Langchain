import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import load_prompt

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the HuggingFaceEndpoint as the underlying LLM
# for the ChatHuggingFace model.
repo_id = "deepseek-ai/DeepSeek-R1" # The model you were trying to use

llm_for_chat = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=100,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Wrap the LLM with ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm_for_chat)

st.title("LLM Prompt UI")
st.header('Research Tool')
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
template = load_prompt("/workspaces/Langchain/Prompt/template.json")

if st.button("summarize"):
    chain = template | chat_model | StrOutputParser()
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result)
