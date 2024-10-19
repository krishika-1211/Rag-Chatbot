import os
import logging
import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pdf_utils import extract_text_from_pdf
from pinecone_utils import store_vectors, query_pinecone, dimension
from models import ChatbotRequest, ChatQuery
from database import get_top_chatbots
from transformers import AutoTokenizer, AutoModel, pipeline
from httpx import ConnectTimeout
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chatbot_router = APIRouter()

# Hugging Face models and configurations
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Change to desired model
HUGGINGFACE_GENERATION_MODEL = "gpt2"  # Or another text-generation model
MAX_TOKEN_LENGTH = 512

# Load models
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_EMBEDDING_MODEL)
model = AutoModel.from_pretrained(HUGGINGFACE_EMBEDDING_MODEL)

# Initialize the Hugging Face text generation pipeline
generator = pipeline('text-generation', model=HUGGINGFACE_GENERATION_MODEL)

# Helper function to generate embeddings using Hugging Face
async def generate_embeddings_with_huggingface(text: str):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    
    # Split the input text into chunks if it exceeds the max token length
    if len(tokens) > MAX_TOKEN_LENGTH:
        chunks = [tokens[i:i + MAX_TOKEN_LENGTH] for i in range(0, len(tokens), MAX_TOKEN_LENGTH)]
        all_embeddings = []

        for chunk in chunks:
            # Convert the chunk into the proper tensor format for the model
            chunk = torch.tensor(chunk).unsqueeze(0)  # Add batch dimension
            
            # Create the attention mask for the chunk
            attention_mask = torch.ones(chunk.shape)

            inputs = {
                'input_ids': chunk,
                'attention_mask': attention_mask
            }

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
            
            all_embeddings.append(embeddings.squeeze().tolist())

        # Averaging the embeddings across all chunks
        final_embedding = torch.mean(torch.tensor(all_embeddings), dim=0).tolist()
    else:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_TOKEN_LENGTH)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
        final_embedding = embeddings.squeeze().tolist()
        
    if len(final_embedding) != dimension:
        raise ValueError(f"Embedding size mismatch: expected {dimension}, got {len(final_embedding)}")
    
    return final_embedding
    
# Helper function to generate a response using Hugging Face
async def generate_with_huggingface(prompt: str):
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

@chatbot_router.post("/create/")
async def create_chatbot(
    pdf: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...),
    prompt: str = Form(...)
):
    logger.info("Creating chatbot with name: %s", name)
    
    pdf_text = extract_text_from_pdf(pdf)

    if not pdf_text:
        logger.error("Failed to extract text from PDF")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

    # Create embeddings using Hugging Face
    try:
        embedding_response = await generate_embeddings_with_huggingface(pdf_text)
        embeddings = embedding_response  # Adjusted since HF output is a tensor
        logger.info("Embeddings created successfully for chatbot: %s", name)
    except Exception as e:
        logger.exception("Error creating embeddings for chatbot: %s", name)
        raise HTTPException(status_code=500, detail="Error creating embeddings")

    # Store embeddings in Pinecone or your vector storage
    try:
        store_vectors(embeddings, prompt)
    except Exception as e:
        logger.exception(f"Error storing vectors for chatbot: {name}")
        raise HTTPException(status_code=500, detail="Error storing vectors")

    logger.info("Vectors stored successfully")
    return {"message": "Vectors stored successfully", "name": name, "description": description, "prompt": prompt}


@chatbot_router.post("/chat/")
async def chat(chat_query: ChatQuery):
    # Generate the embedding for the user's query
    try:
        query_vector = await generate_embeddings_with_huggingface(chat_query.query)
    except Exception as e:
        logger.error("Error generating query vector: %s", e)
        raise HTTPException(status_code=500, detail="Error processing query")

    # Query Pinecone with the generated vector
    try:
        top_matches = query_pinecone(query_vector,top_k=5)
        if not top_matches:
            raise HTTPException(status_code=404, detail="No relevant context found")
    except Exception as e:
        logger.error("Error querying Pinecone: %s", e)
        raise HTTPException(status_code=500, detail="Error querying Pinecone")

    # Extract context from top matches
    context = " ".join([match.get('text', '') for match in top_matches if 'text' in match])

    # Generate a response using Hugging Face
    try:
        prompt = f"Based on the following context: {context}, answer the query: {chat_query.query}"
        hf_response = await generate_with_huggingface(prompt)
        answer = hf_response.strip()
    except Exception as e:
        logger.error("Error generating response from Hugging Face: %s", e)
        raise HTTPException(status_code=500, detail="Error generating response")

    return {"answer": answer, "context": context}

@chatbot_router.get("/list_chatbots/")
async def list_chatbots():
    try:
        top_chatbots = await get_top_chatbots(limit=5)
        logger.info("Listing top chatbots: %s", top_chatbots)
    except Exception as e:
        logger.error("Error fetching top chatbots: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching chatbots")

    return {"top_chatbots": top_chatbots}
