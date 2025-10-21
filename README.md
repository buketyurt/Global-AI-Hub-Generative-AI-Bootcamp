# 🧠 Mental Health Chatbot (RAG-Based)  
**Global AI Hub – Generative AI Bootcamp Project**  
Project Owner: **Buket Yurt**  
Hugging Face Space: [🌐 View Project](https://huggingface.co/spaces/buketyurt/global_ai_hub_generative_ai-bootcamp)

---

## 2- Project Objective  

This project aims to develop a **chatbot that provides answers to mental health–related questions** using the **Retrieval-Augmented Generation (RAG)** architecture.  
The system retrieves information from a predefined FAQ dataset and, when necessary, fetches the most relevant content from an **external vector database** to enhance its responses.  

The main goal is to minimize the "hallucination" problem common in LLMs and provide **accurate, evidence-based answers** supported by retrieved information.  

---

## Dataset Information  

- **Source:** [Mental Health FAQ for Chatbot (Kaggle)](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)  
- **Content:**  
  - Frequently asked questions about topics such as mental health, depression, anxiety, therapy, stress management, and sleep regulation  
  - Each row contains a user question paired with a professionally written answer  

The dataset was directly obtained from Kaggle and required no complex preprocessing.  
However, to ensure optimal model performance, all text was cleaned before embedding — including normalization of lowercase usage, spacing, and punctuation when needed.

---


## Project Architecture (RAG Pipeline)  

The project consists of the following main components:

1. **Data Loading:**  
   Mental health question–answer pairs are loaded from the Kaggle dataset.  
2. **Embedding Generation:**  
   Each text entry is converted into vector space using the `sentence-transformers` model.  
3. **Vector Storage:**  
   The generated embeddings are stored in the `ChromaDB` database.  
4. **Query Processing:**  
   When a user submits a question, it is embedded and compared against the Chroma database using similarity search.  
5. **LLM Answer Generation:**  
   The Gemini model produces a final answer by combining the user’s question with the retrieved knowledge snippets.  

```text
User Query → Embedding → ChromaDB Search → Closest Documents → 
Gemini Model (LLM) → Final Answer
```
## 3- **Operating Instructions**

1. **Installing Requirements**  
   In the project directory:  
   ```bash
   # Create venv, activate, install dependencies
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Defining Environment Variables**  
   Specify your Gemini API key in the `app.py` file:  
   ```bash
   # Example
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

3. **Run**  
   To launch the project in Hugging Face or locally:  
   ```bash
   python app.py
   ```

4. **Accessing the Web Interface**  
   The interface opens via Gradio. Users can ask questions on this screen.


##  4 – Solution Architecture  

This project applies a **Retrieval-Augmented Generation (RAG)** approach to improve the factual accuracy of chatbot responses on mental health topics.  
Instead of relying solely on a large language model, the system **retrieves relevant knowledge** from a structured FAQ dataset and then uses that context to generate an accurate, grounded answer.

---
##  Technologies Used  

| Layer | Technology |
|--------|-------------|
| **LLM (Generative Model)** | Gemini 2.5 Flash (Google) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database** | ChromaDB |
| **RAG Pipeline Framework** | LangChain |
| **Frontend / UI** | Hugging Face Gradio Interface |
| **Environment** | Hugging Face Spaces (Python 3.10) |

---

###  Pipeline Overview  

1. **Data Loading & Cleaning**  
   - The *Mental Health FAQ for Chatbot* dataset from Kaggle is loaded and normalized for consistent formatting.  

2. **Embedding Generation**  
   - Each question–answer pair is embedded using `sentence-transformers/all-MiniLM-L6-v2`, which converts text into vector representations based on semantic meaning.  

3. **Vector Storage**  
   - Embeddings are stored in **ChromaDB**, allowing quick similarity searches by cosine distance.  

4. **Query Retrieval**  
   - A user’s question is embedded and matched with the most relevant documents in ChromaDB.  

5. **Answer Generation**  
   - Retrieved content is combined with the query and passed to **Gemini 2.5 Flash**, which generates a coherent and factual response.  

---

### 🔄 Workflow Summary  

```text
User Question → Embedding → ChromaDB Retrieval → Gemini (LLM) → Answer
```
## 🌐 5 – Web Interface & Product Guide  

You can directly access and test the project via the deployed interface below:  
🔗 **[View Live Demo on Hugging Face](https://huggingface.co/spaces/buketyurt/global_ai_hub_generative_ai-bootcamp)**  

---

### 🧠 Interface Overview  

The **Mental Health FAQ Chatbot** is an interactive web app built with **Gradio**, designed to help users ask and receive informative, evidence-based responses about mental health topics.  
It operates entirely on the browser and connects to the backend RAG pipeline in real time.

Key interface features:
- 💬 **Chat Window:** Users can type or select mental health questions.  
- 🧩 **Memory Context:** The chatbot remembers your session history to give more context-aware responses.  
- 📄 **Suggested Prompts:** Common starter questions like *“What does it mean to have a mental illness?”* or *“How can I manage anxiety?”* appear on the main screen.  
- ⚠️ **Disclaimer:** A reminder states that the chatbot is for informational purposes only and not a replacement for professional mental health advice.  

---

### 🖥️ User Workflow  

1. Open the [Hugging Face Space](https://huggingface.co/spaces/buketyurt/global_ai_hub_generative_ai-bootcamp).  
2. Select or type a question in the chat box.  
3. The system retrieves related context from the vector database.  
4. The Gemini LLM generates a grounded, factual response.  
5. The chatbot displays the answer in the chat window while preserving the session memory.  

---

### 🧩 Example Interaction  

![Chat Example](https://github.com/buketyurt/Global-AI-Hub-Generative-AI-Bootcamp/blob/main/images/Capture.PNG)

---

### 🎬 Live Demo Preview  

Below is a short preview of the chatbot interaction flow:  

![Chatbot Demo](https://github.com/buketyurt/Global-AI-Hub-Generative-AI-Bootcamp/blob/main/images/Global_AI_Hub_Generative_AI_Bootcamp_Full_Slower.gif)

---

The web interface demonstrates how **RAG-based chatbots** can combine retrieval accuracy and generative fluency in a clean, user-friendly layout — ideal for educational or awareness-based AI assistants.
