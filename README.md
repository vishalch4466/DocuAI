# DocuAI

## Overview
DocuAI is an intelligent document analysis platform that allows users to interact with PDF documents through a natural language interface. Upload your PDF files and ask questions directly about their content - DocuAI will provide accurate, contextual answers based on the document information.

## Features
- **PDF Processing**: Upload and process PDF documents of any size and complexity
- **Conversational Interface**: Interact with your documents through a chatbot interface
- **Contextual Understanding**: Get precise answers based on the actual content of your documents
- **Simple Setup**: Easy to configure with your own OpenAI API key

## How It Works
1. Upload your PDF document to the platform
2. Ask questions about the document content in natural language
3. Receive accurate answers extracted directly from the document

## Installation

### Prerequisites
- Python 3.8+
- FastAPI
- OpenAI API key

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/DocuAI.git
cd DocuAI
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key
```
OPENAI_API_KEY=your_api_key_here
```

4. Start the application
```bash
python main.py
```

## Usage
1. Access the web interface at `http://localhost:8000`
2. Upload your PDF file
3. Start asking questions about the document content
4. Receive contextual answers based on the document information

## Architecture
DocuAI uses a FastAPI backend with Python for document processing and connects to OpenAI's API for natural language understanding and response generation.

## License
[MIT License](LICENSE)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.