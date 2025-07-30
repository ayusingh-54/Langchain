from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import load_prompt
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import os 

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# For debugging, you can keep this temporarily:
print(f"DEBUG: Google API Key loaded: {'Yes' if google_api_key else 'No'}")
if not google_api_key:
    print("ERROR: GOOGLE_API_KEY is not set. Please check your .env file.")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.7,
    max_output_tokens=1000, # <--- INCREASED THIS VALUE
    api_key=google_api_key
)

class Review(BaseModel):
    key_themes: List[str] = Field(
        description="Write down all the key themes discussed in the review in a list format"
    )
    summary: str = Field(
        description="Write a summary of the review in a single paragraph"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Write the sentiment of the review as positive, negative or neutral"
    )
    pros: Optional[List[str]] = Field(
        default=None, 
        description="Write down the pros of the product in a list format"
    )
    cons: Optional[List[str]] = Field(
        default=None, 
        description="Write down the cons of the product in a list format"
    )
    name: str = Field(
        description="Write the name of the product"
    ) 

structured_model = model.with_structured_output(Review)

try:
    result = structured_model.invoke("""love this product! It has changed my life for the better. The quality is top-notch and the customer service is excellent. Highly recommend it to anyone looking for a reliable and effective solution.
    The only downside is that it is a bit pricey, but I believe it is worth every penny. Overall, I am extremely satisfied with my purchase and will definitely be a repeat customer. The product is amazing! I have been using it for a few weeks now and I can already see a significant improvement in my daily routine. The design is sleek and modern, making it a great addition to my home. The only issue I have encountered is that the battery life could be better, but it is not a deal breaker for me.""")
    
    if result:
        print(result)
        print(result.key_themes)
        print(result.summary)
    else:
        print("The model did not return a structured output. 'result' is None.")
except Exception as e:
    print(f"An error occurred during model invocation: {e}")