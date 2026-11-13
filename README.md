Energy Operations Document RAG Assistant

**Railway Deploy Link : web-production-73a072f3.up.railway.app**

A Retrieval-Augmented Generation (RAG) assistant that answers questions about renewable-energy operations documents (safety procedures, maintenance SOPs) and cites the source for every answer.

Built as a case study for Greenko.

Problem

Operations teams lose time searching through long SOP documents. Wrong or invented answers in a safety-critical setting are costly. This assistant gives grounded answers taken only from the uploaded documents, and says so when the answer is not in them.

Features
Ask questions about SOP documents in plain language
Every answer includes citations (document name and section/page)
Hallucination checks: if the retrieved context does not support an answer, the assistant declines instead of guessing
FastAPI backend with a simple web interface
Deployed on Railway
How It Works
SOP documents -> chunking -> embeddings -> vector store
                                              |
User question -> embed -> retrieve top-k chunks
                                              |
                       LLM answers using only retrieved chunks
                                              |
                       Answer + citations + grounding check
Ingest: documents are split into chunks and converted to embeddings.
Retrieve: the question is embedded and the most relevant chunks are fetched.
Generate: the LLM answers strictly from those chunks.
Verify: the answer is checked against the retrieved context; unsupported answers are rejected.
Cite: the source chunks are returned with the answer.
Tech Stack
Layer	Technology
Backend	Python, FastAPI, Uvicorn
Retrieval	<vector store, e.g. FAISS / Chroma>
Embeddings	<embedding model>
LLM	<LLM provider / model>
Frontend	<HTML/JS or React>
Deployment	Railway
Project Structure
rag-app/
├── backend/
│   ├── app.py            # FastAPI app and API routes
│   ├── ...               # ingestion, retrieval, generation modules
│   └── requirements.txt
├── frontend/             # UI (if applicable)
├── Procfile              # Railway start command
├── .gitignore
└── README.md
Getting Started
Prerequisites
Python 3.11+
An API key for the LLM/embedding provider
Installation
powershell
git clone https://github.com/<your-username>/rag-app.git
cd rag-app/backend

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
Environment Variables

Create a .env file in backend/ (never commit it):

API_KEY=your_api_key_here
Run Locally

From the backend folder:

powershell
python -m uvicorn app:app --reload

Open http://127.0.0.1:8000 (API docs at http://127.0.0.1:8000/docs).

Usage
Upload or place the SOP documents in the configured documents folder.
Run ingestion to build the index.
Ask a question, for example: "What is the lockout/tagout procedure before turbine maintenance?"
You receive an answer with citations to the source documents.
Deployment (Railway)
Push this repository to GitHub.
In Railway, create a New Project -> Deploy from GitHub repo.
Add the environment variables (for example API_KEY) under Variables.
Railway uses the Procfile to start the app:
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
Open the generated Railway URL.
Limitations and Future Work
Answer quality depends on document quality and chunking strategy
Scanned PDFs may need OCR
Planned: user feedback on answers, multi-document filtering, evaluation dashboard
