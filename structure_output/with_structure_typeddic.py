from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import load_prompt
from typing import List, TypedDict , Annotated , Optional , Literal
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Initialize the HuggingFaceEndpoint as the underlying LLM
# for the ChatHuggingFace model.
repo_id = "deepseek-ai/DeepSeek-R1"  # The model you were trying to use
llm_for_chat = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=100,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
# Wrap the LLM with ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm_for_chat)  
class Review(TypedDict):
    key_themes : Annotated[List[str], "Write down all the key themes discussed in the review in a list format"]
    summary : Annotated[str, "Write a summary of the review in a single paragraph"]
    sentiment : Annotated[Literal["positive", "negative", "neutral"], "Write the sentiment of the review as positive, negative or neutral"]
    pros: Annotated[Optional[List[str]], "Write down the pros of the product in a list format"]
    cons: Annotated[Optional[List[str]], "Write down the cons of the product in a list format"]
    name : Annotated[str, "Write the name of the product"]

structured_model = chat_model.with_structured_output(Review)

result = structured_model.invoke("""
love this product! It has changed my life for the better. The quality is top-notch and the customer service is excellent. Highly recommend it to anyone looking for a reliable and effective solution.
  The only downside is that it is a bit pricey, but I believe it is worth every penny. Overall, I am extremely satisfied with my purchase and will definitely be a repeat customer. The product is amazing! I have been using it for a few weeks now and I can already see a significant improvement in my daily routine. The design is sleek and modern, making it a great addition to my home. The only issue I have encountered is that the battery life could be better, but it is not a deal breaker for me."""
)
print(result)