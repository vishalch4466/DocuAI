import os
import re
import nltk
import faiss
import PyPDF2
import openai

from nltk.tokenize import sent_tokenize
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set
openai.api_key = "sk-"
app = FastAPI()

# In-memory history (replace with a database in production)
chat_history: List[Dict] = []

# -------------------------------------------------------
#               CORS Configuration
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
#               Global Variables & Models
# -------------------------------------------------------
PDF_TEXTS = {}  # filename -> extracted text

# Single model instantiation for embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {str(e)}")

# Single QA pipeline initialization
QA_MODEL_NAME = "deepset/roberta-base-squad2"
try:
    qa_pipeline = pipeline("question-answering", model=QA_MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load QA model '{QA_MODEL_NAME}': {str(e)}")


# -------------------------------------------------------
#               Utility Functions
# -------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Remove extra whitespace, special characters, and page markers from PDF text.
    """
    if not text:
        return ""
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove special characters except basic punctuation
    text = re.sub(r"[^\w\s.,!?]", "", text)
    # Remove page numbers (e.g., 'Page 1' or 'page 2')
    text = re.sub(r"(Page \d+|page \d+)", "", text)
    return text.strip()


def chunk_text_by_sentence(pdf_text: str, chunk_size: int = 512) -> list:
    """
    Split text into chunks by sentence. Each chunk will be up to `chunk_size` tokens (approx).
    """
    if not pdf_text:
        return []

    sentences = sent_tokenize(pdf_text)
    chunks = []
    current_chunk = []
    current_length = 0

    # Roughly count words as a proxy for chunk size
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = word_count
        else:
            current_chunk.append(sentence)
            current_length += word_count

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_faiss_index(chunks: list) -> (faiss.IndexFlatL2, list):
    """
    Create and return a FAISS index, along with the embeddings array.
    """
    if not chunks:
        raise ValueError("No chunks to create FAISS index.")

    # Encode all chunks
    embeddings = sentence_model.encode(chunks, batch_size=16, show_progress_bar=False)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def call_chatgpt(query: str, system_prompt: str = "") -> str:
    """
    Fallback to ChatGPT (gpt-3.5-turbo) when the PDF-based QA is not satisfactory.
    """
    try:
        messages = []
        # Optionally add a system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # User query
        messages.append({"role": "user", "content": query})

        response = openai.ChatCompletion.create(
            # model="gpt-3.5",
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        # Extract the assistant's response
        gpt_answer = response["choices"][0]["message"]["content"]
        return gpt_answer.strip()
    except Exception as e:
        return f"Error calling ChatGPT: {str(e)}"


# -------------------------------------------------------
#               API Endpoints
# -------------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Saves the uploaded PDF temporarily.
    2. Extracts and cleans its text.
    3. Stores the text in PDF_TEXTS dict for future querying.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are allowed."
        )

    # Save the PDF to disk (temporary)
    file_path = f"./{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unable to save file: {str(e)}"
        )

    # Extract text using PyPDF2
    try:
        reader = PyPDF2.PdfReader(file_path)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pdf_text += clean_text(page_text) + " "

        # Store the full text in memory
        PDF_TEXTS[file.filename] = pdf_text.strip()
    except Exception as e:
        # Cleanup file and raise
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Error reading PDF: {str(e)}"
        )
    finally:
        # Remove the file to save space
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"message": "PDF processed successfully", "filename": file.filename}


@app.get("/list")
async def list_uploaded_files():
    """
    Returns a list of filenames that have been uploaded.
    """
    return {"uploaded_files": list(PDF_TEXTS.keys())}


@app.post("/chat")
async def chat(
    query: str,
    filename: str,
    top_k: int = 1,
    threshold: float = 0.2  # <-- Tolerance / Confidence Threshold
):
    """
    1. Find the top_k relevant chunks from the PDF text for the query.
    2. Use a QA pipeline to extract an answer from those chunks.
    3. If the QA pipeline's confidence < threshold or answer is empty -> fallback to ChatGPT.
    """
    # 1) Check if PDF is in memory
    if filename not in PDF_TEXTS:
        # Fallback to ChatGPT if the PDF doesn't exist in memory
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        return {
            "query": query,
            "answer": gpt_answer,
            "score": None,
            "context_snippet": "PDF not found in memory",
            "filename": filename,
            "top_k": top_k
        }

    pdf_text = PDF_TEXTS[filename]
    if not pdf_text:
        # Fallback to ChatGPT if the PDF text is empty
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        return {
            "query": query,
            "answer": gpt_answer,
            "score": None,
            "context_snippet": "Empty PDF text",
            "filename": filename,
            "top_k": top_k
        }

    # 2) Chunk the PDF text
    chunks = chunk_text_by_sentence(pdf_text, chunk_size=512)
    if not chunks:
        # Fallback to ChatGPT if no chunks exist
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        return {
            "query": query,
            "answer": gpt_answer,
            "score": None,
            "context_snippet": "No chunks extracted",
            "filename": filename,
            "top_k": top_k
        }

    # 3) Create FAISS index
    try:
        index, embeddings = create_faiss_index(chunks)
    except Exception as e:
        # Fallback to ChatGPT if index creation fails
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create FAISS index. ChatGPT fallback: {gpt_answer}"
        )

    if index.ntotal == 0:
        # Fallback to ChatGPT if the index is empty
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        return {
            "query": query,
            "answer": gpt_answer,
            "score": None,
            "context_snippet": "FAISS index is empty",
            "filename": filename,
            "top_k": top_k
        }

    # 4) Encode the query
    try:
        query_embedding = sentence_model.encode([query])
    except Exception as e:
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode query. ChatGPT fallback: {gpt_answer}"
        )

    if query_embedding.shape[1] != embeddings.shape[1]:
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        raise HTTPException(
            status_code=500,
            detail="Dimension mismatch between query and chunk embeddings. "
                   f"ChatGPT fallback: {gpt_answer}"
        )

    # 5) Search in the FAISS index
    try:
        distances, indices = index.search(query_embedding, top_k)
    except Exception as e:
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        raise HTTPException(
            status_code=500, detail=f"FAISS search failed. ChatGPT fallback: {gpt_answer}"
        )

    if len(indices[0]) == 0:
        # Fallback to ChatGPT if no relevant chunks found
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        return {
            "query": query,
            "answer": gpt_answer,
            "score": None,
            "context_snippet": "No relevant chunks found",
            "filename": filename,
            "top_k": top_k
        }

    # Combine the top_k chunks
    combined_context = " ".join(chunks[idx] for idx in indices[0] if idx >= 0)

    # 6) Run QA pipeline on the combined context
    try:
        qa_result = qa_pipeline(question=query, context=combined_context)
        answer = qa_result.get("answer", "").strip()
        score = qa_result.get("score", 0.0)
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print(score)
        print("++++++++++++++++++++++++++++++++++++++++++++++++")

        # Check if the answer meets our threshold
        # or if it's empty/"none"
        if not answer or answer.lower() in ["", "none"] or score < threshold:
            # Fallback to ChatGPT
            gpt_answer = call_chatgpt(query)
            chat_history.append({
                "user": query,
                "bot": gpt_answer,
                "score": None,
                "context_snippet": combined_context,
                "source": "ChatGPT (fallback)"
            })
            return {
                "query": query,
                "answer": gpt_answer,
                "score": None,
                "context_snippet": combined_context,
                "filename": filename,
                "top_k": top_k
            }

        # Otherwise, trust the QA pipeline's answer
        chat_history.append({
            "user": query,
            "bot": answer,
            "score": score,
            "context_snippet": combined_context,
            "source": "PDF QA"
        })
        return {
            "query": query,
            "answer": answer,
            "score": score,
            "context_snippet": combined_context,
            "filename": filename,
            "top_k": top_k
        }

    except Exception as e:
        # Fallback to ChatGPT on any QA pipeline error
        gpt_answer = call_chatgpt(query)
        chat_history.append({"user": query, "bot": gpt_answer, "source": "ChatGPT"})
        raise HTTPException(
            status_code=500, detail=f"QA pipeline error. ChatGPT fallback: {gpt_answer}"
        )


@app.get("/history")
async def get_history():
    """
    Return the conversation history from memory.
    """
    return {"history": chat_history}


# -------------------------------------------------------
#               Run with Uvicorn (for local testing)
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Run the API locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
