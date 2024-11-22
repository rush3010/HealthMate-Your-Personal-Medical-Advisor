# **HealthMate Chatbot**

## **Overview**
HealthMate is a medical chatbot designed to assist users with medical queries by extracting relevant information from PDF documents. It uses state-of-the-art natural language processing (NLP) techniques and the LLaMA2 quantized model to deliver accurate and concise answers based on the provided data. The chatbot is built using the LangChain framework and integrates a variety of libraries for document processing, embeddings, vector databases, and conversational interfaces.

---

## **Features**
- **Custom PDF Data Integration**: Loads medical information from PDF files as the knowledge base.
- **Conversational Interface**: Provides a user-friendly chatbot interface via Chainlit.
- **Efficient Query Handling**: Uses advanced sentence embeddings to retrieve the most relevant information.
- **Vector Database Support**: Leverages FAISS for fast and efficient search over large datasets.
- **Scalable NLP Framework**: Built on LangChain to easily integrate other data sources or language models.
- **LLaMA2 Model**: Employs the LLaMA2 quantized model for efficient, resource-friendly natural language understanding.

---

## **Libraries and Tools Used**
1. **[LangChain](https://github.com/hwchase17/langchain)**  
   - Framework for building applications powered by language models.
2. **[Chainlit](https://chainlit.io/)**  
   - Tool for creating conversational user interfaces with Python backends.
3. **[Sentence-Transformers](https://www.sbert.net/)**  
   - Generates embeddings for semantic search and retrieval.
4. **[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)**  
   - A library for efficient similarity search and clustering of dense vectors.
5. **[PyPDF2](https://github.com/py-pdf/pypdf)**  
   - Used for reading and processing PDF documents.
6. **[LLaMA2 Quantized Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)**  
   - Efficient, lightweight version of LLaMA2 for natural language processing tasks.

---

## **How It Works**
1. **Data Preparation**:
   - PDF files containing medical data are loaded and processed.
   - Text is split into manageable chunks for better query handling using LangChain’s `RecursiveCharacterTextSplitter`.

2. **Embedding Generation**:
   - Each chunk is converted into dense vector embeddings using the `sentence-transformers` library.

3. **Vector Database Creation**:
   - The embeddings are stored in a FAISS vector database for fast retrieval during queries.

4. **LLaMA2 Model Integration**:
   - The LLaMA2 quantized model is used as the language model to process user queries and provide responses based on the retrieved information.

5. **Conversational Interface**:
   - The chatbot is deployed via Chainlit, offering a seamless interaction experience for users.

---
