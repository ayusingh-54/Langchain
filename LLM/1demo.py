from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)


response = model.invoke("capital of india ?")
print(response.content)