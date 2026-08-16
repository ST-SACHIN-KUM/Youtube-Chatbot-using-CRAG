# YouTube Chatbot with RAG and CRAG

This project is a YouTube chatbot that allows users to ask questions about a video by providing its URL. It also includes a Chrome extension for a smoother browser-based experience.

The chatbot combines **RAG (Retrieval-Augmented Generation)** and **CRAG (Corrective Retrieval-Augmented Generation)** concepts using **LangGraph** to define the data flow and control how information moves through the pipeline.

## Overview

The system works by taking a YouTube video URL, extracting the transcript, splitting it into chunks, embedding the chunks, and retrieving the most relevant context for a user query. The retrieved context is then passed to an LLM to generate a grounded answer.

To improve reliability, the project also implements **CRAG**, which adds a corrective step to evaluate retrieved documents and refine the context before generation. This helps reduce irrelevant retrievals and improves answer quality.

## Key Features

- YouTube video URL input.
- Transcript extraction from video.
- Document chunking and embedding.
- Vector-based retrieval for relevant context.
- RAG-based answer generation.
- CRAG-based correction and refinement.
- LangGraph workflow for state-based control flow.
- Chrome extension integration for easy access in the browser.

## RAG Workflow

The RAG pipeline follows these steps:

1. Accept a YouTube video URL and user question.
2. Extract the transcript from the video.
3. Split the transcript into smaller chunks.
4. Convert chunks into embeddings and store them in a vector database.
5. Retrieve the most relevant chunks for the query.
6. Combine the retrieved context with the question.
7. Send the prompt to the LLM.
8. Return the final answer.
<img width="146" height="427" alt="Screenshot 2026-07-21 112824" src="https://github.com/user-attachments/assets/83fd546a-7530-4b2a-9bd8-549bdc3e9b19" />

## CRAG Workflow

The CRAG pipeline improves upon basic RAG by adding a correction step:

1. Retrieve documents for the user query.
2. Evaluate whether the retrieved context is relevant.
3. If retrieval quality is low, rewrite or refine the query/context.
4. Filter out irrelevant chunks.
5. Generate the final answer using the corrected context.

This makes the chatbot more robust when retrieval is noisy or incomplete.

<img width="457" height="547" alt="Screenshot 2026-07-21 113054" src="<img width="1171" height="540" alt="Screenshot 2026-08-16 161539" src="https://github.com/user-attachments/assets/c6b3bbb5-20fa-4db7-a16f-90c562690d92" />
" />

## LangGraph Flow

The application uses **LangGraph** to manage the workflow as a graph. Each node represents a step in the pipeline, and each edge defines how the data moves from one step to another.

Typical nodes include:

- `refine_document` – improves the retrieved context if needed.
- `generate_answer` – creates the final answer using the refined context.

The graph allows the chatbot to dynamically decide what happens next based on the current state.

## State Structure

The graph uses a shared state object to pass data between nodes.

Example state:

```python
class State(TypedDict):
    input: str
    docs: list
    refine_doc: list
    answer: str
```

Each node reads from the state and returns updated values, which are then merged into the workflow.

## Chrome Extension

A Chrome extension is included so users can interact with the chatbot directly from the browser while watching a YouTube video.

The extension:

- Reads the current YouTube video URL.
- Sends the video ID and user query to the backend.
- Displays the chatbot response in a popup UI.

## Technologies Used

- Python
- Streamlit
- LangChain
- LangGraph
- RAG / CRAG
- Chroma or similar vector database
- Chrome Extension
- Groq / Gemini / other LLMs

## Conclusion

This project combines a YouTube chatbot, RAG, CRAG, and LangGraph into one workflow-driven system. The graph-based design makes the pipeline modular, easier to debug, and easier to extend with new retrieval and correction steps.
