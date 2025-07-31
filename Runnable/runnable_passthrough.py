from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough ,RunnableParallel , RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

# Define the HuggingFace endpoint
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following topic: {topic}",
    input_variables=["topic"]
)
output_parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt1, model, output_parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, output_parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))
