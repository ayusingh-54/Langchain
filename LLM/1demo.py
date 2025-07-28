from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)


response = model.invoke("Hello, what is the weather like in Surat, Gujarat today?")
print(response.content)