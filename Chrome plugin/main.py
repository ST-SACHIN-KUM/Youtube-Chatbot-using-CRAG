from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re
import streamlit as st
from langchain_groq import ChatGroq
import json
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

with open('file.json', 'r') as file:
    data = json.load(file)
    
app = FastAPI()

# CORS: allow all origins (OK for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_api_key = data['GROQ_API_KEY']


os.environ["GROQ_API_KEY"] = groq_api_key
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    )
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=data['GEMINI_API_KEY']
)


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "youtube_videos"

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    fetched_transcript = api.fetch(video_id, languages=["en"])
    transcript_data = fetched_transcript.to_raw_data()
    transcript = " ".join(chunk["text"] for chunk in transcript_data)
    
    return transcript


def ensure_video_indexed(video_id: str) -> None:

    collection = vector_store._collection
    existing = collection.get(where={"video_id": video_id}, include=[])
    if existing["ids"]:
        return  # already indexed

    transcript = get_transcript(video_id)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={"video_id": video_id})
        for chunk in chunks
    ]
    vector_store.add_documents(docs)

def build_rag_chain(video_id: str):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5, "filter": {"video_id": video_id}}
    )

    qa_prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering about a YouTube video. "
        "Use ONLY the following retrieved context from the video transcript. "
        "If the answer is not in the context, say so explicitly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

class ChatRequest(BaseModel):
    video_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

# ----------------- Endpoint -----------------

@app.post("/api/chat")
def chat(req: ChatRequest) -> ChatResponse:
    # Ensure this video is indexed in Chroma
    ensure_video_indexed(req.video_id)

    # Build RAG chain for this video
    rag_chain = build_rag_chain(req.video_id)

    # Get answer
    answer = rag_chain.invoke(req.question)

    return ChatResponse(answer=answer)