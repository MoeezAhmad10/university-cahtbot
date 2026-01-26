# 🎓 Riphah International University Chatbot

An intelligent AI-powered chatbot that provides comprehensive information about Riphah International University using RAG (Retrieval-Augmented Generation) technology. The chatbot scrapes official Riphah University websites, creates a searchable knowledge base, and answers questions about programs, admissions, fees, campuses, and more.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Streamlit UI](#streamlit-ui)
  - [FastAPI Backend](#fastapi-backend)
  - [Command Line Interface](#command-line-interface)
- [API Documentation](#api-documentation)
- [How It Works](#how-it-works)
- [Caching System](#caching-system)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- **🤖 AI-Powered Responses**: Uses OpenAI's GPT-3.5-turbo for intelligent, context-aware answers
- **🔍 RAG Architecture**: Retrieval-Augmented Generation ensures accurate, source-based responses
- **🌐 Web Scraping**: Automatically scrapes 40+ official Riphah University web pages
- **💾 Smart Caching**: Vector store caching for faster initialization and reduced API costs
- **📱 Multiple Interfaces**: 
  - Streamlit web UI for interactive chat
  - FastAPI REST API for integration with other applications
  - Command-line interface for direct terminal use
- **📚 Source Attribution**: Optional source URL display for transparency
- **🔄 Data Refresh**: Manual refresh capability to update with latest university information
- **🎯 Contextual Understanding**: Enhanced question processing for better retrieval
- **🏫 Multi-Campus Support**: Covers Islamabad, Lahore, Faisalabad, and Sahiwal campuses

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ├─── Streamlit UI (chatbotui.py)
       ├─── FastAPI REST API (api.py)
       └─── CLI (chatbot.py)
       │
       ▼
┌──────────────────────────────────┐
│   RiphahChatbot Core Engine      │
│  ┌────────────────────────────┐  │
│  │  Web Scraper (40+ URLs)    │  │
│  └────────┬───────────────────┘  │
│           ▼                       │
│  ┌────────────────────────────┐  │
│  │  Document Processing       │  │
│  │  (Text Chunking)           │  │
│  └────────┬───────────────────┘  │
│           ▼                       │
│  ┌────────────────────────────┐  │
│  │  OpenAI Embeddings         │  │
│  │  (text-embedding-3-small)  │  │
│  └────────┬───────────────────┘  │
│           ▼                       │
│  ┌────────────────────────────┐  │
│  │  FAISS Vector Store        │  │
│  │  (Similarity Search)       │  │
│  └────────┬───────────────────┘  │
│           ▼                       │
│  ┌────────────────────────────┐  │
│  │  GPT-3.5-turbo             │  │
│  │  (Response Generation)     │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

## 🛠️ Technologies Used

### Core AI/ML Stack
- **LangChain**: Framework for building LLM applications
- **OpenAI API**: GPT-3.5-turbo for chat, text-embedding-3-small for embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector storage and retrieval

### Web & API
- **FastAPI**: High-performance async REST API
- **Streamlit**: Interactive web UI framework
- **Uvicorn**: ASGI server for FastAPI
- **BeautifulSoup4**: Web scraping and HTML parsing

### Data Processing
- **LangChain Text Splitters**: Document chunking
- **Python-dotenv**: Environment variable management
- **NumPy**: Numerical computations

## 📁 Project Structure

```
ripah_chatbot/
├── chatbot.py              # Core chatbot logic & CLI interface
├── chatbotui.py            # Streamlit web UI application
├── api.py                  # FastAPI REST API server
├── check.py                # Environment variable checker utility
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── chatbot_cache/          # Cached vector stores by date
│   ├── riphah_vectorstore_20260122/
│   │   └── index.faiss
│   └── ...
└── documents/              # Reference PDFs (optional)
    ├── Faculty of Computing.pdf
    ├── Riphah International University (RIU) Overview.pdf
    └── Riphah_University_CS_Lahore_Overview.pdf
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **OpenAI API Key**: Required for embeddings and chat completions
- **Internet Connection**: For web scraping and API calls
- **Minimum 2GB RAM**: For vector store operations
- **Disk Space**: ~100MB for dependencies, ~50MB for cache

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ripah_chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python check.py
```

This should display whether your OpenAI API key is properly configured.

## ⚙️ Configuration

### 1. Create `.env` File

Create a file named `.env` in the project root directory:

```env
OPENAI_API_KEY=sk-proj-your-openai-api-key-here
```

### 2. Get OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy and paste into your `.env` file

**Important**: Never commit your `.env` file to version control!

## 💻 Usage

### Streamlit UI

The easiest way to interact with the chatbot:

```bash
streamlit run chatbotui.py
```

This will:
- Open a browser window at `http://localhost:8501`
- Provide a beautiful, user-friendly chat interface
- Show conversation history
- Display source URLs when available

**Features**:
- Full-page chat interface
- Message history
- Real-time responses
- Responsive design
- Source attribution toggle

### FastAPI Backend

Run the REST API server for programmatic access:

```bash
python api.py
```

Or with Uvicorn directly:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

**API will be available at**:
- API Base: `http://127.0.0.1:8000`
- Interactive Docs: `http://127.0.0.1:8000/docs`
- Alternative Docs: `http://127.0.0.1:8000/redoc`

### Command Line Interface

Run the chatbot directly in your terminal:

```bash
python chatbot.py
```

**CLI Commands**:
- `sources` - Ask a question and see source URLs
- `refresh` - Re-scrape all websites and update data
- `clear` - Clear conversation history
- `save` - Save conversation to JSON file
- `quit` - Exit the chatbot

## 📡 API Documentation

### Endpoints

#### `GET /`
Root endpoint with API information.

**Response**:
```json
{
  "message": "Riphah University Chatbot API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "chat": "/chat",
    "refresh": "/refresh",
    "docs": "/docs"
  }
}
```

#### `GET /health`
Check API and chatbot health status.

**Response**:
```json
{
  "status": "healthy",
  "chatbot_initialized": true,
  "timestamp": "2026-01-26T10:30:00"
}
```

#### `POST /chat`
Main chat endpoint for asking questions.

**Request Body**:
```json
{
  "question": "What programs are available at Riphah?",
  "include_sources": false
}
```

**Response**:
```json
{
  "answer": "Riphah International University offers...",
  "sources": ["https://riphah.edu.pk/programs/"],
  "timestamp": "2026-01-26T10:30:00",
  "success": true
}
```

#### `POST /refresh`
Refresh chatbot data by re-scraping all websites (runs in background).

**Response**:
```json
{
  "message": "Data refresh started in background",
  "success": true,
  "timestamp": "2026-01-26T10:30:00"
}
```

#### `GET /campuses`
Get list of all Riphah campuses with URLs.

**Response**:
```json
{
  "campuses": [
    {"name": "Islamabad Campus", "url": "https://riphah.edu.pk/islamabad-campus/"},
    {"name": "Lahore Campus", "url": "https://riphah.edu.pk/lahore-campus/"},
    {"name": "Faisalabad Campus", "url": "https://riphahfsd.edu.pk/"},
    {"name": "Sahiwal Campus", "url": "https://riphahsahiwal.edu.pk/"}
  ],
  "main_website": "https://riphah.edu.pk/"
}
```

#### `GET /quick-links`
Get important quick links for Riphah University.

**Response**:
```json
{
  "links": {
    "main_website": "https://riphah.edu.pk/",
    "programs": "https://riphah.edu.pk/programs/",
    "admissions": "https://riphah.edu.pk/admissions/",
    "fee_structure": "https://riphah.edu.pk/fee-structure/",
    "scholarships": "https://riphah.edu.pk/scholarships/",
    "contact": "https://riphah.edu.pk/contact/"
  }
}
```

### Example API Usage

#### Using cURL

```bash
# Health check
curl http://127.0.0.1:8000/health

# Ask a question
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the fee structure for Computer Science?", "include_sources": true}'

# Get campuses
curl http://127.0.0.1:8000/campuses
```

#### Using Python Requests

```python
import requests

# Ask a question
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "question": "What are the admission requirements for MBBS?",
        "include_sources": True
    }
)
print(response.json())
```

#### Using JavaScript/Fetch

```javascript
// Ask a question
fetch('http://127.0.0.1:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'Tell me about scholarships at Riphah',
    include_sources: true
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🔍 How It Works

### 1. **Data Collection**
- Scrapes 40+ official Riphah University web pages
- Covers all campuses, programs, admissions, fees, facilities, and student services
- Extracts clean text content using BeautifulSoup4

### 2. **Document Processing**
- Splits web content into 800-character chunks with 150-character overlap
- Creates manageable segments for embedding and retrieval
- Preserves context across chunk boundaries

### 3. **Vector Embedding**
- Converts text chunks into numerical vectors using OpenAI's `text-embedding-3-small`
- Creates semantic representations that capture meaning
- Stores vectors in FAISS index for fast similarity search

### 4. **Query Processing**
When you ask a question:
1. **Question Enhancement**: Adds contextual keywords for better retrieval
2. **Semantic Search**: Finds 4 most relevant document chunks using vector similarity
3. **Context Assembly**: Combines retrieved chunks into coherent context
4. **LLM Generation**: GPT-3.5-turbo generates answer based on context and system prompt
5. **Response Delivery**: Returns formatted answer with optional source URLs

### 5. **Caching System**
- Saves processed vector stores to disk with date stamps
- Reuses cached data on subsequent runs (same day)
- Dramatically reduces initialization time and API costs

## 💾 Caching System

The chatbot implements an intelligent caching system to optimize performance:

### Cache Location
```
chatbot_cache/
└── riphah_vectorstore_YYYYMMDD/
    ├── index.faiss          # Vector index
    └── index.pkl            # Document metadata
```

### Cache Behavior
- **Automatic**: Cache is automatically created after first data load
- **Date-Based**: New cache created daily (format: `YYYYMMDD`)
- **Reusable**: Cached data is loaded on subsequent runs within the same day
- **Benefits**:
  - Instant initialization (vs. 5-10 minutes for fresh scrape)
  - Reduced OpenAI API costs
  - Offline capability (after initial cache creation)

### Manual Cache Management
```python
# Use cached data (default)
chatbot.initialize(use_cache=True)

# Force fresh data scrape
chatbot.initialize(use_cache=False)

# Refresh data during runtime
chatbot.refresh_data()
```

## 🌐 Data Sources

The chatbot scrapes information from:

### Main Website
- Homepage, About, Vision/Mission
- University introduction and values

### Academic Information
- All faculties and departments
- Complete program listings
- Faculty profiles

### Admissions & Fees
- General admission process
- Program-specific admission requirements
- Comprehensive fee structures
- Tuition fees by program

### Campus-Specific
- **Islamabad Campus**: Main campus information
- **Lahore Campus**: Programs and facilities
- **Faisalabad Campus**: Complete website (riphahfsd.edu.pk)
- **Sahiwal Campus**: Complete website (riphahsahiwal.edu.pk)

### Student Services
- Scholarship programs
- Financial aid
- Hostel accommodation
- Library facilities
- Campus life information

### Research & Innovation
- Research initiatives
- ORIC (Office of Research, Innovation & Commercialization)
- Innovation hub

### Administrative
- Quality assurance
- Accreditation
- Contact information
- News and events

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
Error: OPENAI_API_KEY not found in .env file!
```
**Solution**: Create `.env` file with your OpenAI API key

#### 2. Import Errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

#### 3. FAISS Loading Errors
```
RuntimeError: Error in faiss::FileIOReader
```
**Solution**: Delete cached vectorstore and reinitialize

#### 4. Slow Initialization
**Solution**: Wait for initial scraping to complete (~5-10 mins). Subsequent runs use cache and are instant.

#### 5. Rate Limit Errors
```
openai.error.RateLimitError
```
**Solution**: 
- Wait a few minutes before retrying
- Check your OpenAI usage limits
- Use cached data to avoid re-embedding

### Debug Mode

Use `check.py` to verify your environment:

```bash
python check.py
```

Expected output:
```
OPENAI_API_KEY from environment: ✓ FOUND
First 10 chars: sk-proj-xx
```

## 🎯 Example Questions

The chatbot can answer questions like:

**Admissions**:
- "What are the admission requirements for Computer Science?"
- "How do I apply to Riphah University?"
- "What documents are needed for MBBS admission?"

**Programs**:
- "What undergraduate programs are available?"
- "Tell me about the MBA program"
- "Which campuses offer Computer Science?"

**Fees**:
- "What is the fee structure for BS Computer Science?"
- "How much does the MBBS program cost?"
- "Are there any scholarships available?"

**Campus Life**:
- "Tell me about hostel facilities"
- "What services are available for students?"
- "Where is the Faisalabad campus located?"

**General**:
- "What is Riphah University's mission?"
- "How many campuses does Riphah have?"
- "What research facilities are available?"

## 📊 Performance Metrics

- **Initial Load Time**: 5-10 minutes (first run, web scraping)
- **Cached Load Time**: 2-5 seconds (subsequent runs)
- **Average Response Time**: 2-4 seconds per query
- **Documents Processed**: 40+ web pages
- **Vector Store Size**: ~1,500-2,000 text chunks
- **Embedding Model**: text-embedding-3-small (1536 dimensions)

## 🔒 Security Notes

- **Never commit `.env` file** to version control
- **API Key Protection**: Keep your OpenAI API key secret
- **CORS Configuration**: In production, restrict API origins in `api.py`
- **Rate Limiting**: Consider implementing rate limits for production use
- **Authentication**: Add authentication for `/refresh` endpoint in production

## 🔄 Data Refresh Strategy

### Automatic
- Cache expires daily (based on date stamp)
- New cache created automatically when date changes

### Manual
- **CLI**: Type `refresh` command
- **Streamlit**: Use refresh button in sidebar
- **API**: Call `POST /refresh` endpoint

**When to Refresh**:
- University updates programs or fees
- New campus information is added
- Admission requirements change
- At the start of each academic session

## 📈 Future Enhancements

Potential improvements for the chatbot:

- [ ] Multi-language support (Urdu)
- [ ] PDF document parsing from `/documents` folder
- [ ] Conversation history persistence
- [ ] User feedback collection
- [ ] Advanced analytics and usage tracking
- [ ] Integration with Riphah's official portal
- [ ] Voice input/output capabilities
- [ ] Mobile application
- [ ] WhatsApp/Telegram bot integration
- [ ] Fine-tuned model for Riphah-specific queries
- [ ] Automated daily data refresh
- [ ] Email notification system for admissions
- [ ] Comparison tool for different programs

## 📝 Development Notes

### Key Classes and Methods

**RiphahChatbot Class** (`chatbot.py`):
- `__init__(openai_api_key, cache_dir)`: Initialize chatbot
- `initialize(custom_urls, use_cache)`: Load data from cache or web
- `chat(question)`: Get answer to a question
- `chat_with_sources(question)`: Get answer with source URLs
- `refresh_data()`: Re-scrape all websites
- `load_specific_urls(urls)`: Scrape specific URLs
- `process_documents(documents)`: Create vector embeddings

### Configuration Parameters

**Embedding Model**:
- Model: `text-embedding-3-small`
- Dimension: 1536
- Cost: ~$0.02 per 1M tokens

**LLM Model**:
- Model: `gpt-3.5-turbo`
- Temperature: 0.7 (balanced creativity/accuracy)
- Max Tokens: 300 (concise responses)

**Text Splitter**:
- Chunk Size: 800 characters
- Chunk Overlap: 150 characters
- Length Function: `len()`

**Retriever**:
- Search Type: Similarity
- Top K: 4 documents per query

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is created for educational purposes for Riphah International University.

## 👥 Author

Developed with ❤️ for Riphah International University

## 🙏 Acknowledgments

- **Riphah International University** for providing comprehensive online resources
- **OpenAI** for GPT and embedding models
- **LangChain** for the excellent RAG framework
- **Facebook AI** for FAISS vector search
- **Streamlit** for the beautiful UI framework

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact Riphah IT department
- Visit [Riphah Official Website](https://riphah.edu.pk/)

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Active Development
