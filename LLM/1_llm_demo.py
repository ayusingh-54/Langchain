import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Load the .env file from the parent directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Try loading with different encodings to handle BOM and other encoding issues
try:
    load_dotenv(env_path, encoding='utf-8-sig')
except:
    try:
        load_dotenv(env_path, encoding='utf-8')
    except:
        load_dotenv(env_path)

# Also try loading from current directory as fallback
load_dotenv()

# Manually get the key from the environment and configure it
# This FORCES the library to use your key.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print(f"Looking for .env file at: {env_path}")
        print(f".env file exists: {os.path.exists(env_path)}")
        exit()
    
    print(f"API Key found: {api_key[:10]}..." if api_key else "No API key found")
    
    # Set the environment variable explicitly for LangChain
    os.environ["GOOGLE_API_KEY"] = api_key
    
    genai.configure(api_key=api_key)
    print("API Key configured successfully! ✅")
except Exception as e:
    print(f"Could not find or configure API Key. Error: {e} ❌")
    # Exit if key is not found
    exit()

# Now, initialize the model with the API key
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

try:
    result = model.invoke("what is the capital of France?  ")
    print(result)
except Exception as e:
    print(f"Error invoking model: {e}")