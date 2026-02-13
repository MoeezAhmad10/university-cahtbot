# 🎓 Riphah University Chatbot

AI-powered chatbot for Riphah International University information using RAG (Retrieval-Augmented Generation). Provides accurate answers about programs, admissions, fees, campuses, and more.

## ✨ Features

- 🤖 **Intelligent Responses** - Multi-stage reasoning with GPT-3.5-turbo
- 🔍 **RAG Architecture** - Retrieves information from 40+ official university pages
- 💾 **Smart Caching** - Fast initialization with daily vector store caching
- 📱 **Multiple Interfaces** - Streamlit UI, FastAPI, and CLI
- 🗂️ **Session Management** - Conversation context tracking
- 📊 **Monitoring** - Performance metrics and logging
- 🛡️ **Rate Limiting** - API abuse protection

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ripah_chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file in project root:

```env
OPENAI_API_KEY=sk-proj-your-api-key-here
```

Get your API key from [OpenAI Platform](https://platform.openai.com/)

### 3. Run

**Streamlit UI** (Recommended):
```bash
streamlit run chatbotui.py
```
Opens at `http://localhost:8501`

**FastAPI Server**:
```bash
python api.py
```
API: `http://127.0.0.1:8000` | Docs: `http://127.0.0.1:8000/docs`

**Command Line**:
```bash
python chatbot.py
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | System health and metrics |
| `/chat` | POST | Ask questions (with session support) |
| `/session/{id}` | GET | Get session info |
| `/session/{id}/clear` | DELETE | Clear session history |
| `/metrics` | GET | Performance statistics |
| `/refresh` | POST | Refresh data (background) |
| `/campuses` | GET | List campuses |
| `/quick-links` | GET | Important links |

### Example API Usage

```bash
# Ask a question
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What programs does Riphah offer?", "include_sources": true}'

# Check health
curl http://127.0.0.1:8000/health

# View metrics
curl http://127.0.0.1:8000/metrics
```

## 📁 Project Structure

```
ripah_chatbot/
├── chatbot.py        # Core logic & CLI
├── chatbotui.py      # Streamlit UI
├── api.py            # FastAPI server
├── requirements.txt  # Dependencies
├── .env             # API keys (create this)
├── chatbot_cache/   # Vector store cache
└── documents/       # Optional local docs
```

## 💻 Usage Examples

### Streamlit UI
- Interactive chat interface
- Conversation history
- Source attribution
- Session management

### FastAPI

**Python**:
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "What are MBBS admission requirements?"}
)
print(response.json())
```

**JavaScript**:
```javascript
fetch('http://127.0.0.1:8000/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: 'Tell me about scholarships'})
})
.then(res => res.json())
.then(data => console.log(data));
```

### CLI Commands
```bash
sources  # Ask with source URLs
refresh  # Update data
clear    # Clear history
save     # Save conversation
quit     # Exit
```

## 🎯 Example Questions

- "What programs are available at Riphah?"
- "What are admission requirements for Computer Science?"
- "What is the fee structure for MBBS?"
- "Tell me about the Faisalabad campus"
- "What scholarships are available?"
- "How do I apply for admission?"

## 🔧 Troubleshooting

**API Key Error**:
```bash
python check.py  # Verify API key setup
```

**Module Not Found**:
```bash
pip install -r requirements.txt
```

**Slow First Run**:
- Initial scraping takes 5-10 minutes
- Subsequent runs use cache (2-5 seconds)

**Cache Issues**:
- Delete `chatbot_cache/` folder and reinitialize

## 📊 Key Technologies

- **LangChain** - RAG framework
- **OpenAI** - GPT-3.5-turbo & embeddings
- **FAISS** - Vector similarity search
- **FastAPI** - REST API
- **Streamlit** - Web UI
- **Pydantic** - Data validation

## 🔒 Security

- Never commit `.env` file
- Keep API keys secret
- Configure CORS for production
- Add authentication for admin endpoints

## 📝 Environment Variables

```env
OPENAI_API_KEY=sk-proj-xxx        # Required
API_HOST=127.0.0.1                # Optional (default: 127.0.0.1)
API_PORT=8000                     # Optional (default: 8000)
API_RELOAD=true                   # Optional (default: true)
DEBUG=false                       # Optional (hide errors in prod)
```

## 🎓 Data Coverage

- **Campuses**: Islamabad, Lahore, Faisalabad, Sahiwal
- **Content**: Programs, admissions, fees, facilities, scholarships, research
- **Sources**: 40+ official Riphah web pages
- **Update**: Manual refresh or daily cache expiration



**Version**: 2.0.0 | **Last Updated**: February 2026 | **Status**: Production Ready
