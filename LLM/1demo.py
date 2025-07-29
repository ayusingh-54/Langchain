import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

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

# Define a chat prompt template using message types
# Chat models expect input as a list of message objects.
# Even for a single-turn question, it's best to frame it as a conversation.
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of {country}?")
    ]
)

# Create a chain that uses the chat model
chain = chat_template | chat_model | StrOutputParser()

# Invoke the chain with conversational input
country_name = "Germany"
response = chain.invoke({"country": country_name})
print(f"The capital of {country_name} is: {response}\n")

country_name = "India"
response = chain.invoke({"country": country_name})
print(f"The capital of {country_name} is: {response}\n")

# You can also directly invoke with a list of messages:
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="What is the capital of France?")
# ]
# direct_response = chat_model.invoke(messages)
# print(f"Direct chat response: {direct_response.content}")