from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
import os 

google_api_key = os.getenv("GOOGLE_API_KEY")

#

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.7,
    max_output_tokens=1000, # <--- INCREASED THIS VALUE
    api_key=google_api_key
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

