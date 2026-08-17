from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import TypedDict
from langchain_groq import ChatGroq
from google import genai
from google.genai import types
import streamlit as st
from langchain_core.documents import Document
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import json
from groq import Groq
import re
import os


with open('file.json', 'r') as file:
    data = json.load(file)
    

def extract_video_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    if re.match(r"^[a-zA-Z0-9_-]{11}$", s):
        return s
    # Try to extract from URL
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)
    return None

def get_transcript(video_id: str) -> str | None:
    try:
        # 1. Instantiate the API
        api = YouTubeTranscriptApi()

        # 2. Fetch the transcript
        fetched_transcript = api.fetch(video_id, languages=["en"])

        # 3. Extract the text, start times and length
        transcript_data = fetched_transcript.to_raw_data()

        # Convert it into plain raw text
        transcript = " ".join(chunk["text"] for chunk in transcript_data)
        
        return transcript
    
    except TranscriptsDisabled:
        return None

st.set_page_config(page_title="YouTube Summarizer", layout="centered")
st.title("YouTube Summarizer")

llm = None
embeddings = None
groq_api_key = data['GROQ_API_KEY']


os.environ["GROQ_API_KEY"] = groq_api_key
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    )
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key = data['GEMINI_API_KEY']
)

lower_bound = data["LOWER_BOUND"]
upper_bound = data["UPPER_BOUND"]

CHROMA_PATH = "./chroma_db_2"
COLLECTION_NAME = "youtube_videos"

if embeddings is not None:
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
else:
    vector_store = None


prompt_2 = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def get_score(query, document):
  relevance_prompt = ChatPromptTemplate.from_template(
    "You are evaluating how relevant a document is for a given query.\n\n"
    "Query: {query}\n"
    "Document: {document}\n\n"
    "Rate the relevance on a scale from 0 to 1, where:\n"
    "0 = completely irrelevant\n"
    "1 = perfectly relevant and sufficient to answer the query.\n\n"
    "Return only a single float (e.g., 0.85)."
    )
  relevance_chain = relevance_prompt | llm | StrOutputParser()
  score_str = relevance_chain.invoke({"query": query, "document": document})
  score = float(score_str.strip())
  return score    

def web_search(query):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer in 1–3 short sentences. "
                    "Give only the key facts, no explanations. "
                    "Use up-to-date web info if needed."
                    ),
                },
            {
                "role": "user",
                "content": query,
            },
            ],
        )
    return response.choices[0].message.content

def clarifying(query):
  relevance_prompt = ChatPromptTemplate.from_template(
    "You are evaluating whether a given query is ambigous or not.\n\n"
    "Query: {query}\n"
    "Rate the ambiguity on a scale from 0 to 1, where:\n"
    "0 = completely correct, no ambiguity\n"
    "1 = perfectly ambigous query.\n\n"
    "Return only a single float (e.g., 0.85)."
    )
  relevance_chain = relevance_prompt | llm | StrOutputParser()
  score_str = relevance_chain.invoke({"query": query})
  score = float(score_str.strip())
  return score

rewrite_prompt = ChatPromptTemplate.from_template(
    "You are rewriting an question to be different in wording but keep the same meaning.\n"
    "Question: {question}\n"
    "Do not repeat the previous wording.\n"
    "Use a new structure, different vocabulary, and a fresh style.\n"
    "Keep it concise, relevant, and factually correct.\n"
)


class State(TypedDict):
    input : str
    docs : list
    refine_docs : list
    route : str
    feedback : str
    ambiguous : int
    clarification : str
    final_answer : str

def ambiguous_score(state : State):
    query = state["input"]
    score = clarifying(query)
    return {"ambiguous" : score}
    
def route_query(state : State):
    score = state["ambiguous"]
    if score > 0.6:
        return {"route":"clarify"}
    return {"route" : "answer"}
    
def clarify_question(state : State):
    return {"clarification" : "Your question is ambiguous. Can you clarify what you want?"}
    
def get_refine_document(document, query):
  score = get_score(query, document)
  sentence = document.split('.')
  ans = ""

  #Correct document
  if score >= upper_bound:        
    for i in sentence:
      strip_score = get_score(query, i)
      if strip_score > 0.7:
        ans = ans+i
  
  #Ambigous document
  elif score >= lower_bound and score <= upper_bound:
    for i in sentence:
      strip_score = get_score(query, i)
      if strip_score > 0.7:
        ans = ans+i
    web_result = web_search(query)
    ans = ans+web_result
  
  #Incorrect document
  else:
    web_result = web_search(query)
    ans = ans+web_result

  return ans

def retrieve_correct_document(state : State):
    query = state["input"]
    docs = state["docs"]
    final_doc = list()
    for document in docs:
        doc = get_refine_document(document.page_content, query)
        final_doc.append(doc)
    return {"refine_docs" : final_doc}

def generator(state : State):
  context_text = "\n\n".join(doc for doc in state["refine_docs"])
  final_prompt = prompt_2.invoke({"context": context_text, "question": state["input"]})
  answer_1 = llm.invoke(final_prompt)
  return {"answer": answer_1.content}

def web_answer(state : State):
    ans  = web_search(state["input"])
    return {"final_answer" : ans}
    
def human_response(state : State):
    user_input = interrupt({
        "message": "Is this answer good enough?",
        "options" : ["Yes", "llm", "RAG"]
    })
    choice = user_input.get("choice", False)
    
    if choice == "approve":
        return Command(goto="end")
    elif choice == "llm":
        return Command(goto="web_answer")
    else:
        return Command(goto="rewrite_question")
            

def rewrite_question(state : State):
  query = state["input"]
  prompt_chain = rewrite_prompt | llm | StrOutputParser()
  update_question = prompt_chain.invoke({"quesion" : query})

  return {"input" : update_question}

workflow = StateGraph(State)

workflow.add_node("ambiguous_score", ambiguous_score)
workflow.add_node("routing", route_query)
workflow.add_node("clarify", clarify_question)
workflow.add_node("retrieve", retrieve_correct_document)
workflow.add_node("generate_answer", generator)
workflow.add_node("clarify", clarify_question)
workflow.add_node("human_response", human_response)
workflow.add_node("web_answer", web_answer)
workflow.add_node("rewrite_question", rewrite_question)


workflow.add_edge(START, "ambiguous_score")
workflow.add_edge("ambiguous_score", "routing")
workflow.add_conditional_edges("routing", 
                               lambda state: state["route"],
                               {
                                   "clarify" : "clarify",
                                   "answer" : "retrieve"
                               })
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", "human_response")
workflow.add_conditional_edges("human_response", human_response,
                               {
                                   "end" : END,
                                   "web_answer" : "web_answer",
                                   "rewrite_question" : "rewrite_question"
                               })
workflow.add_edge("web_answer", "human_response")
workflow.add_edge("rewrite_question", "routing")
workflow.add_edge("clarify", START)

app = workflow.compile(checkpointer=InMemorySaver())

def get_all_videos() -> list[dict]:

    if vector_store is None:
        return []

    # Get underlying Chroma collection
    collection = vector_store._collection

    # Get all documents + metadata
    result = collection.get(include=["metadatas"])

    ids = result["ids"]
    metadatas = result["metadatas"] or []

    # Build set of video_ids with their display_name
    video_map = {}
    for _, meta in zip(ids, metadatas):
        vid = meta.get("video_id")
        if not vid:
            continue
        name = meta.get("display_name") or vid
        video_map[vid] = name

    return [{"video_id": vid, "display_name": name} for vid, name in video_map.items()]

def update_video_display_name(video_id: str, display_name: str):
    
    if vector_store is None:
        return

    collection = vector_store._collection

    # Get all ids for this video_id
    result = collection.get(
        where={"video_id": video_id},
        include=[],
    )
    ids_to_update = result["ids"]

    if not ids_to_update:
        return

    # Update metadata for each id
    for doc_id in ids_to_update:
        collection.update(
            ids=[doc_id],
            metadatas=[{"video_id": video_id, "display_name": display_name}],
        )
        
def index_video(video_id: str, transcript: str, display_name: str | None = None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(transcript)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "display_name": display_name or video_id,
            },
        )
        for chunk in chunks
    ]
    vector_store.add_documents(docs)
    
st.subheader("Add a new YouTube video")

new_video_input = st.text_input(
    "YouTube video ID or URL",
    key="new_video_input",
    placeholder="e.g. dQw4w9WgXcQ or https://www.youtube.com/watch?v=dQw4w9WgXcQ",
)

new_video_display_name = st.text_input(
    "Display name (optional, you can edit later)",
    key="new_video_display_name",
    placeholder="e.g. 'Intro to RAG - Prof. X'",
)

if st.button("Add Transcript to list"):
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not new_video_input:
        st.error("Please provide a video ID or URL.")
    elif vector_store is None:
        st.error("Vector store not initialized (check API key).")
    else:
        video_id = extract_video_id(new_video_input)
        if not video_id:
            st.error("Invalid YouTube video ID or URL.")
        else:
            with st.spinner("Fetching transcript..."):
                transcript = get_transcript(video_id)
            if not transcript:
                st.warning("Could not fetch transcript. Check if captions are enabled.")
            else:
                with st.spinner("Indexing video..."):
                    index_video(video_id, transcript, new_video_display_name or None)
                st.success(f"Video '{new_video_display_name or video_id}' indexed.")
                
st.subheader("Videos List")

all_videos = get_all_videos()

if not all_videos:
    st.info("No indexed videos yet. Add one above.")
else:
    # Dropdown to select video
    video_options = {v["display_name"]: v["video_id"] for v in all_videos}
    selected_display_name = st.selectbox(
        "Select a video",
        options=list(video_options.keys()),
        key="selected_video_display",
    )

    selected_video_id = video_options[selected_display_name]
    
    # Edit display name
    new_name = st.text_input(
        "Rename this video (display name)",
        value=selected_display_name,
        key="rename_video",
    )
    if new_name != selected_display_name:
        if st.button("Save new name"):
            update_video_display_name(selected_video_id, new_name)
            st.success("Display name updated. Please refresh the page or re-select the video.")
            st.rerun()

    st.write(f"Selected video ID: `{selected_video_id}`")
    
    # Fetch transcript for selected video (for summary tab)
    with st.spinner("Loading transcript for selected video..."):
        transcript = get_transcript(selected_video_id)

    if not transcript:
        st.warning("Could not fetch transcript for this video.")
    elif llm is None or vector_store is None:
        st.info("Enter Groq API key to enable summarization and chat.")
    else:
        tab_summary, tab_chat = st.tabs(["Summary", "Chat"])
        
        with tab_summary:
            st.subheader(f"Summary: {selected_display_name}")

            summary_prompt = ChatPromptTemplate.from_template(
                "Summarize the following YouTube transcript in 5–7 bullet points.\n\n"
                "Transcript:\n{transcript}"
            )

            summary_chain = (
                RunnablePassthrough()
                | summary_prompt
                | llm
                | StrOutputParser()
            )
            if st.button("Generate Summary", key="gen_summary_btn"):
                try:
                    with st.spinner("Generating summary..."):
                        summary = summary_chain.invoke({"transcript": transcript})
                        st.write(summary)
                except Exception as e:
                    st.error(f"Error during summarization: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # ----- Chat Tab (RAG with vector DB) -----
        with tab_chat:
            
            thread_id = selected_video_id       #thread has been initialised for each video transcript for persistence in memory
            config={"configurable": {"thread_id": thread_id}}
            
            st.subheader(f"Chat: {selected_display_name}")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Clear chat when changing video
            if "last_video_id" not in st.session_state:
                st.session_state.last_video_id = None
            if st.session_state.last_video_id != selected_video_id:
                st.session_state.messages = []
                st.session_state.last_video_id = selected_video_id
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Retriever scoped to current video
            base_retriever = vector_store.as_retriever(search_type="mmr",
                search_kwargs={"k": 5, "filter": {"video_id": selected_video_id}}
            )
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
                )
            
            if prompt := st.chat_input("Ask something about this video..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        
                        docs = retriever.invoke(prompt)
                        
                        input = {"input":prompt, "docs": docs, "refine_doc":[], "final_answer":"",
                                 "route":"", "feedback":"", "ambiguous":-1, "clarification":""}
                        
                        result = app.invoke(input, config=config)
                        
                        if result["clarification"] != "":
                            st.write(result["clarification"])
                        else:
                            user_choice = st.radio("Choose one", ["approve", "LLM", "RAG"])
                            feedback = st.text_input("Feedback")
                            
                            resume_input = {
                                                "approved": user_choice == "approve",
                                                "feedback": feedback,
                                                "choice": user_choice,
                                            }
                            result = app.invoke(Command=(resume=), config = config)
                        
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result["answer"]}
                        )


