from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)   
prompt2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)
output_parser = StrOutputParser()
chain = RunnableSequence(prompt1 , model, output_parser , prompt2, model, output_parser)
print(chain.invoke({'topic':'AI'}))