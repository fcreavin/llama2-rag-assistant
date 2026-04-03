# Hybrid LLM Solution: RAG + Llama2 Research Assistant

## Overview
This project implements a hybrid large language model (LLM) application combining Retrieval-Augmented Generation (RAG) with a locally hosted Llama2 model using Ollama.

The system enables users to query the Llama 2 research paper through an interactive Streamlit interface, providing context-aware and grounded responses.

---

## Architecture

The system consists of:

- Streamlit UI (chat interface)
- Document processing pipeline (PDF → chunks → embeddings)
- Chroma vector database (retrieval)
- Prompt construction layer
- Ollama-hosted Llama2 model

![Architecture Diagram](images/presentation_architecture_diagram.png)

---

## Features

- Context-aware question answering
- Retrieval grounding (reduces hallucination)
- Local LLM execution (no API required)
- Interactive chat UI
- Source transparency (retrieved excerpts shown)

---

## Installation

### 1. Clone the repository

git clone https://github.com/fcreavin/llama2-rag-assistant.git
cd llama2-rag-assistant

2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
Run the Application
Start Ollama (if not already running)
ollama serve
Run Streamlit
python -m streamlit run RAG_withUI.py
Model Configuration

The application uses:

MODEL_NAME = "AIresearcher"

A fine-tuned model was provided and successfully packaged into Ollama but failed at runtime. Therefore, the stable base model is used in the final implementation.

Fine-Tuned Model (Important Note)

The instructor-provided fine-tuned model file (.bin) is not included in this repository because it exceeds GitHub’s file size limits.

To use the fine-tuned model:

Download the model file separately from the course materials or Hugging Face
Place it in the following directory:
/fine-tunedModel/
Update the Modelfile to reference the local model path
Create the model in Ollama:
ollama create AIresearcher-finetuned -f Modelfile

Note: In this implementation, the fine-tuned model could be successfully created but failed at runtime, so it is not used in the final application.

Document Source

The system processes the Llama 2 research paper located in:

/documents/
Future Improvements
Stabilize fine-tuned model integration
Persistent vector storage
Enhanced prompt engineering
UI improvements
Author

Fred Creavin

Course: Leveraging Llama2 for Advanced AI Solutions
