from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.schema import Document
from langchain.docstore import InMemoryDocstore
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, StoppingCriteriaList
import logging
import matplotlib.pyplot as plt
import re
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can specify a list of origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Ensure the static directory exists
os.makedirs('static', exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define input request model
class QueryRequest(BaseModel):
    query: str
    filters: list[str]

# Load the FAISS index and documents
try:
    faiss_index = faiss.read_index("faiss_index.index")
    with open("output_with_recommendations_legend.json", "r") as f:
        docs_data = json.load(f)
    documents = [Document(page_content=doc_data.get("page_content", "")) for doc_data in docs_data]
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/Phi-3-mini-4k-instruct")
    ft_model = PeftModel.from_pretrained(model, "phi3_finetuned_final4").to("cuda")
except Exception as e:
    logging.error(f"Error loading FAISS index or documents: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading FAISS index or documents: {e}")

# Define embedding function
def embedding_function(contents):
    return [embedding_model.encode(content) for content in contents]

# Define custom stopping criteria class
class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequences, tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_sequence in self.stop_sequences:
            if stop_sequence in decoded_text:
                return True
        return False

# Define function to retrieve values based on the product name only
def retrieve_values(query, faiss_index, docstore, index_to_docstore_id, embedding_function, num_results=5):
    query_embedding = embedding_function([query])[0]
    _, doc_ids = faiss_index.search(np.array([query_embedding]), len(documents))  # Get all results

    unique_vendors = {}
    similar_documents = []

    for doc_id in doc_ids[0]:
        doc = docstore.search(str(index_to_docstore_id[doc_id]))
        if query.lower() in doc.page_content.lower():  # Ensure the product matches the query
            vendor_info = doc.page_content.split(' sold by ')[-1].split(' located ')[0]
            if vendor_info not in unique_vendors:
                unique_vendors[vendor_info] = doc
                similar_documents.append(doc)
            if len(similar_documents) == num_results:
                break

    return similar_documents

# Define function to load and retrieve documents
def load_and_retrieve(query, filters):
    similar_documents = retrieve_values(query, faiss_index, docstore, index_to_docstore_id, embedding_function)
    if not similar_documents:
        raise ValueError("No similar documents found.")
    similar_docs_str = "\n".join([f"'{doc.page_content}'" for doc in similar_documents])
    return similar_documents, similar_docs_str

# Define function to generate and save bar charts
def generate_and_save_bar_charts(data):
    prices = []
    shipping_costs = []
    delivery_times = []
    labels = []

    data_lines = data.split("\n")
    for line in data_lines:
        price_match = re.search(r'priced at (\d+)', line)
        shipping_match = re.search(r'shipping cost of (\d+)', line)
        delivery_match = re.search(r'delivery time of approximately (\d+)', line)
        vendor_match = re.search(r'sold by ([^ ]+)', line)

        if price_match and shipping_match and delivery_match and vendor_match:
            prices.append(int(price_match.group(1)))
            shipping_costs.append(int(shipping_match.group(1)))
            delivery_times.append(int(delivery_match.group(1)))
            labels.append(vendor_match.group(1))

    # Generate bar charts
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].bar(labels, prices, color='skyblue')
    axs[0].set_title('Price Comparison')
    axs[0].set_ylabel('Price')

    axs[1].bar(labels, shipping_costs, color='lightgreen')
    axs[1].set_title('Shipping Cost Comparison')
    axs[1].set_ylabel('Shipping Cost')

    axs[2].bar(labels, delivery_times, color='salmon')
    axs[2].set_title('Delivery Time Comparison')
    axs[2].set_ylabel('Delivery Time (days)')

    plt.tight_layout()

    # Save the image to a file
    image_path = 'static/charts.png'
    plt.savefig(image_path)
    plt.close(fig)
    return image_path

@app.post("/retrieve")
async def retrieve(query_request: QueryRequest):
    try:
        documents, data = load_and_retrieve(query_request.query, query_request.filters)
        instruction_map = {
            "low price": "display the low price vendor and say why should we choose him",
            "shipping cost": "display the vendor with the lowest shipping cost and say why should we choose him",
            "delivery time": "display the vendor with the fastest delivery time and say why should we choose him",
            "default": "display the best vendor based on low price, shipping cost, and delivery time and say why should we choose him"
        }
        instruction = instruction_map.get(query_request.filters[0] if query_request.filters else "default")
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Given 5 input statements separated by comma compare the 5 inputs and display which one can be recommended to the user write a response that is min 100 words appropriately completes the request a some description about him and why should we choose him.
        Instruction:
        {}
        Top 5 matching vendors:
        {}
        """
        formatted_prompt = alpaca_prompt.format(
            instruction,
            data,
        )
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

        # Create stopping criteria
        stop_sequences = ["### Response", "Follow-up Question"]
        stopping_criteria = StoppingCriteriaList([StopOnSequence(stop_sequences, tokenizer)])
        
        output = model.generate(
            **inputs,
            max_new_tokens=370,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            stopping_criteria=stopping_criteria
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)

        # Ensure consistent extraction of the solution part
        response_start = result.find("Solution:")
        if response_start != -1:
            result = result[response_start:].strip()
            # Only take up to the first complete response
            for stop_seq in stop_sequences:
                result = result.split(stop_seq)[0].strip()

        # Clean up the result to remove any duplicate "Top 5 matching vendors"
        if "Top 5 matching vendors:" in result:
            result_parts = result.split("Top 5 matching vendors:")
            if len(result_parts) > 2:
                result = result_parts[0] + "Top 5 matching vendors:" + result_parts[1]

        image_path = generate_and_save_bar_charts(data)
        
        return {"result": result, "top_5_vendors": [doc.page_content for doc in documents], "chart_image_url": image_path}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chart")
async def get_chart():
    return FileResponse('static/charts.png')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
