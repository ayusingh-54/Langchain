from langchain.schema.runnable import RunnableSequence , RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

load_dotenv()
# Load environment variab  les
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template= ' Write a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template= ' write a LinkedIn post about {topic}',
    input_variables=['topic']
)
output_parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet_chain': RunnableSequence(prompt1, model, output_parser),
    'linkedin_chain': RunnableSequence(prompt2, model, output_parser)
})
print(parallel_chain.invoke({'topic': 'AI'}))