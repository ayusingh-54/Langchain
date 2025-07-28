from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
response = model.invoke("Hello, what is the weather like in Surat, Gujarat today?")
print(response.content)