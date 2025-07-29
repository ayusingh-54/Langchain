from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

try:
    # Use OpenAI instead of HuggingFace for better reliability
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=100
    )

    chat_history = [
        SystemMessage(content='You are a helpful AI assistant')
    ]

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break
        
        chat_history.append(HumanMessage(content=user_input))
        
        try:
            result = model.invoke(chat_history)
            print("AI:", result.content)
            chat_history.append(AIMessage(content=result.content))
        except Exception as e:
            print(f"Error generating response: {e}")
            continue

    print("\nChat History:")
    for message in chat_history:
        if hasattr(message, 'content'):
            print(f"{message.__class__.__name__}: {message.content}")

except Exception as e:
    print(f"Error initializing model: {e}")
    print("Please check your OpenAI API key and internet connection.")