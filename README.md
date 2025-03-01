Below is a sample GitHub README for your project based on the code you provided. It assumes the project is named "DocuMind," a Retrieval-Augmented Generation (RAG) based chatbot for querying PDF documents. Feel free to customize it further based on your specific needs!

---

# DocuMind - RAG-based Chat with PDF

DocuMind is a Streamlit-based web application that allows users to upload PDF files, process them, and ask questions based on their content using a Retrieval-Augmented Generation (RAG) pipeline. It leverages Google's Generative AI (Gemini) for embeddings and language model capabilities, combined with LangChain and Chroma for efficient document retrieval and question-answering.

## Features
- Upload single or multiple PDF files.
- Split and process PDF content into manageable chunks.
- Create a vector store for retrieval using Chroma and Google Generative AI embeddings.
- Query the uploaded documents conversationally with an AI assistant powered by Google's Gemini model.
- Persistent chat history within the session.

## Tech Stack
- **Python**: Core programming language.
- **Streamlit**: Frontend for the web application.
- **LangChain**: Framework for building RAG pipelines and managing document retrieval.
- **Google Generative AI**: Provides embeddings (`embedding-001`) and LLM (`gemini-1.5-flash`).
- **Chroma**: Vector database for storing and retrieving document embeddings.
- **PyPDFLoader**: For loading and processing PDF files.

## Prerequisites
Before running the project, ensure you have the following:
- Python 3.8 or higher installed.
- A Google API key for accessing the Gemini models (set as an environment variable `GEMINI_API_KEY`).
- A `.env` file in the project root with your API key:
  ```
  GEMINI_API_KEY=your-google-api-key-here
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KaanSezen1923/documind.git
   cd documind
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a `.env` file (see Prerequisites).

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Launch the app using the command above.
2. Open your browser to `http://localhost:8501`.
3. Upload one or more PDF files via the sidebar.
4. Click "Submit and Process" to process the files and create a retriever.
5. Enter your question in the chat input box and receive answers based on the PDF content.

## Project Structure
```
documind/
├── app.py              # Main Streamlit application file
├── requirements.txt    # List of Python dependencies
├── .env                # Environment variables (API keys)
├── chroma_db/          # Directory for Chroma vector store persistence
└── README.md           # This file
```

## Dependencies
Install the required libraries by running:
```bash
pip install pysqlite3 streamlit python-dotenv langchain langchain-community langchain-google-genai langchain-chroma
```

## How It Works
1. **PDF Processing**: Uploaded PDFs are read using `PyPDFLoader`, split into chunks with `RecursiveCharacterTextSplitter`, and stored as documents.
2. **Embedding & Retrieval**: The document chunks are embedded using Google's `embedding-001` model and stored in a Chroma vector store.
3. **Querying**: User queries are processed by a RAG chain combining retrieval (via Chroma) and generation (via Gemini `gemini-1.5-flash`).
4. **Chat Interface**: Streamlit manages the chat UI and session state for conversation history.

## Example
![Ekran görüntüsü 2025-03-01 160804](https://github.com/user-attachments/assets/4e61230a-ece0-43b1-be85-076abba883c4)

- Upload a PDF about "transformers arthitecture".

  ![image](https://github.com/user-attachments/assets/feac876e-c4ff-42c1-ba76-dd833bcfc1ce)

- Ask: "What is transformers arthitecture ?"

  ![Ekran görüntüsü 2025-03-01 160742](https://github.com/user-attachments/assets/588fa395-81cc-4e4e-ac1f-d3abeeb7f951)

- DocuMind retrieves relevant sections from the PDF and provides an answer.

## Limitations
- Requires a valid Google API key with access to the Gemini models.
- Processing large PDFs may take time depending on system resources.
- Answers are limited to the content of uploaded PDFs or the model's general knowledge if no context is found.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [Streamlit](https://streamlit.io/).
- Powered by [Google Generative AI](https://cloud.google.com/ai/generative-ai).

---

