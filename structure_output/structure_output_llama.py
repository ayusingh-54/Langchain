from  dotenv import load_dotenv
from typing import List, Optional, Literal
import os
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Initialize the HuggingFaceEndpoint as the underlying LLM
# for the ChatHuggingFace model.
repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
huggingface_endpoint = HuggingFaceEndpoint(
    api_token=HUGGINGFACEHUB_API_TOKEN,
    repo_id=repo_id,
    task="text-generation",
    temperature=0.7,    
)

model = ChatHuggingFace(
    llm=huggingface_endpoint,
    max_new_tokens=1000,  # Increased this value
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

result = structured_model.invoke("""love this product! It has changed my life for the better. The quality is top-notch and the customer service is excellent. Highly recommend it to anyone looking for a reliable and effective solution.
The only downside is that it is a bit pricey, but I believe it is worth every penny. Overall, I am extremely satisfied with my purchase and will definitely be a repeat customer. The product is amazing! I have been using it for a few weeks now and I can already see a significant improvement in my daily routine. The design is sleek and modern, making it a great addition to my home. The only issue I have encountered is that the battery life could be better, but it is not a deal breaker for me.""")

print(result)