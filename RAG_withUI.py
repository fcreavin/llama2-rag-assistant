import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama


PDF_PATH = "./documents/Llama2_Open_Foundation_and_Fine-Tuned_Chat_ Models.pdf"
MODEL_NAME = "AIresearcher"
#MODEL_NAME = "AIresearcher-finetuned"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 4


st.set_page_config(
    page_title="Jake - Hybrid Llama2 RAG Assistant",
    page_icon="🦙",
    layout="wide"
)

st.title("🦙 Jake - Hybrid Llama2 RAG Assistant")
st.write(
    "Ask questions about the full Llama 2 research paper using retrieval-augmented generation."
)
st.info("First question may take a little longer while the document is loaded and indexed.")

with st.sidebar:
    st.header("Project Settings")
    st.write(f"**PDF Source:** `{PDF_PATH}`")
    st.write(f"**LLM Model:** `{MODEL_NAME}`")
    st.write(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
    st.write(f"**Chunk Size:** `{CHUNK_SIZE}`")
    st.write(f"**Chunk Overlap:** `{CHUNK_OVERLAP}`")
    st.write(f"**Top-K Retrieval:** `{TOP_K}`")

    st.markdown("---")
    st.markdown("### System Type")
    st.write("Hybrid LLM (RAG + Specialized Model)")

    st.markdown("---")
    st.subheader("Suggested Questions")
    st.markdown(
        """
- What are the main contributions of the Llama 2 paper?
- How does Llama 2 improve training efficiency?
- What safety methods are described in the paper?
- What model sizes are included in Llama 2?
- How does Llama 2 compare to earlier open models?
"""
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner=False)
def build_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_llm():
    return Ollama(model=MODEL_NAME)


def ask_question(user_question: str):
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()

    retrieved_docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are Jake, an AI research assistant specializing in the Llama 2 paper.

Use the retrieved context below to answer the user's question.
If the answer is not supported by the context, say that clearly.
Be concise, accurate, and easy to understand.

Format your answer like this:
1. Start with a short direct answer.
2. Then provide 2-5 bullet points if they help explain the answer.
3. Only use information supported by the retrieved context.

Context:
{context}

Question:
{user_question}
"""

    answer = llm.invoke(prompt)
    return answer, retrieved_docs


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            st.caption("Response generated using retrieved document context + Llama2 model")
            with st.expander("View retrieved source excerpts"):
                for i, doc in enumerate(msg["sources"], start=1):
                    page_num = doc.metadata.get("page", "Unknown")
                    source_name = doc.metadata.get("source", "PDF")
                    excerpt = doc.page_content[:700].strip()

                    st.markdown(f"**Source {i}**")
                    st.caption(f"Document: {source_name} | Page: {page_num}")
                    st.write(excerpt + "...")


prompt = st.chat_input("Ask a question about the Llama 2 paper...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Jake is reading the paper and generating a response..."):
            try:
                answer, sources = ask_question(prompt)

                st.markdown("### Answer")
                st.write(answer)
                st.caption("Response generated using retrieved document context + Llama2 model")

                with st.expander("View retrieved source excerpts"):
                    for i, doc in enumerate(sources, start=1):
                        page_num = doc.metadata.get("page", "Unknown")
                        source_name = doc.metadata.get("source", "PDF")
                        excerpt = doc.page_content[:700].strip()

                        st.markdown(f"**Source {i}**")
                        st.caption(f"Document: {source_name} | Page: {page_num}")
                        st.write(excerpt + "...")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"### Answer\n\n{answer}",
                        "sources": sources
                    }
                )

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )