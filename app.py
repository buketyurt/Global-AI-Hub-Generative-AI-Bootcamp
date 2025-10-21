import os
import gradio as gr
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GEMINIAPIKEY') or os.environ.get('GOOGLE_API_KEY')

print(f"Environment variables available: {list(os.environ.keys())}")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    print("⚠️ GEMINI_API_KEY not found")
    print("Please add it in Space Settings → Variables and secrets")

# Load the pre-built Chroma database
CHROMA_DIR = Path('./mental_health_chroma')

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

MODEL_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.95,
    'top_k': 40,
    'max_output_tokens': 2048,
}

# Load vector store
try:
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ Vector database loaded successfully!")
except Exception as e:
    print(f"❌ Error loading vector database: {e}")
    retriever = None

# RAG function WITH chat history support
def rag_answer(query, chat_history=None, k=5, model_name="gemini-2.0-flash"):
    """Generate answer using RAG pipeline with chat history"""
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not configured. Please add it to Hugging Face Secrets."
    
    if retriever is None:
        return "❌ Vector database not available. Please check the setup."
    
    try:
        # Retrieved relevant documents
        ctx_docs = retriever.invoke(query)[:k]
        context = "\n\n".join([f"[Doc {i+1}]\n{d.page_content}" for i, d in enumerate(ctx_docs)])
        
        # System prompt
        system = (
            "You are a helpful mental health support assistant. "
            "Use ONLY the provided context to answer questions. "
            "If the answer is not in the context, say you don't know. "
            "Be empathetic, supportive, and concise in your responses. "
            "Consider the conversation history when responding."
        )
        
        # Builded chat history string
        history_text = ""
        if chat_history:
            for user_msg, bot_msg in chat_history:
                history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        # Created full prompt with history
        prompt = f"""{system}
[CONVERSATION HISTORY]
{history_text if history_text else "No previous conversation."}
[CONTEXT FROM DATABASE]
{context}
[CURRENT QUESTION]
{query}
[INSTRUCTIONS]
Provide a helpful answer based on the context and conversation history. If referring to previous messages, be natural about it."""
        
        # Generate response
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=MODEL_CONFIG
        )
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Gradio chat function WITH history
def chat_function(message, history):
    """Handle chat interactions with history"""
    if not message or message.strip() == "":
        return "Please enter a message."
    
    # Pass the history to RAG function
    response = rag_answer(message.strip(), chat_history=history)
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_function,
    title="🧠 Mental Health FAQ Chatbot (with Memory)",
    description="""
    This chatbot uses RAG (Retrieval Augmented Generation) to answer mental health related questions.
    It remembers your conversation history within the same session.
    
    **Note:** This is for informational purposes only and not a substitute for professional mental health advice.
    """,
    examples=[
        "What does it mean to have a mental illness?",
        "How can I manage anxiety?",
        "What are common symptoms of depression?",
        "Who does mental illness affect?",
        "How can I find help?",
        "What are some coping strategies for stress?",
        "What causes mental illness?",
        "How can I support a friend with mental health issues?"
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True)
