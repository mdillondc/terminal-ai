# Samantha AI Assistant

![Samantha AI Assistant Screenshot](screenshot.png)

A terminal AI assistant with advanced features including RAG (Retrieval-Augmented Generation), web search, TTS, and more.

## Features

- **Multi-Provider AI**: OpenAI models, Google Gemini, and Ollama (local models)
- **RAG System**: Query your own documents with intelligent retrieval
- **Web Search**: Real-time information via Tavily API
- **YouTube Integration**: Extract and analyze video transcripts
- **Text-to-Speech**: Convert responses to natural speech
- **Conversation Management**: Save, resume, and organize conversations
- **Rich Commands**: Extensible command system with tab completion
- **Clipboard Integration**: Use clipboard content as input

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (required)
- Tavily API key (optional, for web search)
- Google API key (optional, for Gemini models)
- Ollama (optional, for local models and private RAG)

### Setup

1. **Set API keys**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export TAVILY_API_KEY="your-tavily-key"
   export GOOGLE_API_KEY="your-google-key"
   ```
   
2. **Clone and set up environment**
   ```bash
   git clone <repository-url>
   cd samantha
   conda create -n samantha python=3.10
   conda activate samantha
   pip install -r requirements.txt
   ```
3. **Run**
   ```bash
   python src/main.py
   ```

## Usage

### Basic Commands

```bash
# Switch AI models
> --model gpt-4o-mini

# Enable web search
> --search

# Apply instruction templates
> --instructions summary.md

# Enable text-to-speech
> --tts
> --tts-voice nova
> --tts-save-as-mp3                # Save audio files locally

# Analyze YouTube video
> --youtube https://youtube.com/watch?v=example

# Use clipboard content
> --cb
```

### RAG (Document Analysis)

```bash
# Create document collection (supports .txt and .md files)
mkdir -p rag/my-docs
mkdir -p rag/my-docs/api          # Subdirectories supported
cp your-files.txt rag/my-docs/
cp api-docs/*.md rag/my-docs/api/

# Build and activate collection
> --rag-build my-docs
> --rag my-docs

# Query your documents
> What are the main topics in my documents?
> --rag-show filename.txt         # View specific file chunks
> --rag-status                    # Check RAG configuration
> --rag-test                      # Test embedding provider
```

### Embedding Providers

**OpenAI** (cloud-based):
- High quality, requires API key

**Ollama** (local, private):
- Free, runs locally, works offline
- Install: `curl -fsSL https://ollama.com/install.sh | sh`
- Setup: `ollama serve && ollama pull nomic-embed-text`

Configure in `src/settings_manager.py`:
```python
self.embedding_provider = "ollama"  # or "openai"
```

### Conversation Management

```bash
# Resume previous conversation
# AI will suggest log names automatically
> --log name-of-conversation.md

# Rename current conversation with AI suggested title
> --logmv

# Rename current conversation with your own title
> --logmv project-planning-discussion

# Private mode (no logging)
> --incognito
```

### Getting Help

```bash
> --help                   # Show all commands
> --refresh-models         # Update available models
> --rag-status             # Show RAG system status
> --rag-info               # Show embedding model details
```

## Configuration

### Instruction Templates
Create custom instruction files in `instructions/`:

### Settings
Key settings in `src/settings_manager.py`:
- Model selection and API configuration
- RAG parameters (chunk size, top-k results, embedding provider)
- TTS voice and model settings (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`)
- Cache duration for model listings
- Audio file saving (experimental `--tts-save-as-mp3` feature)

### Audio Files
When `--tts-save-as-mp3` is enabled, audio files are saved as:
```
tts_{timestamp}_{text_preview}.mp3
```

## File Structure

```
samantha/
├── src/                   # Application source code
├── instructions/          # Instruction templates
├── rag/                   # Document collections
├── logs/                  # Conversation logs
├── cache/                 # Model and completion cache
└── requirements.txt       # Python dependencies
```

## Dependencies

See requirements.txt

## API Costs & Privacy

- **OpenAI**: Pay per token
- **Tavily**: Free tier available, paid plans for higher usage
- **Google**: Free tier available
- **Ollama**: Free (runs locally, fully private)

**Privacy Note**: Use Ollama for sensitive documents - data never leaves your machine.

## Quick Start Example

```bash
python src/main.py --input "Hello Samantha!"

# Enable features and start chatting
> --tts --search
> What's the latest news about AI?

# Work with documents (private with Ollama)
> --rag-build my-research
> --rag my-research
> Summarize the key findings from my research papers

# Configure for privacy in src/settings_manager.py:
# self.embedding_provider = "ollama"
# self.ollama_embedding_model = "nomic-embed-text"
```

## RAG Collections & Source Citations

RAG responses include source citations:
```
**Sources:**
• config.md (lines 12-18, relevance: 89%)
• readme.txt (lines 45-52, relevance: 76%)
```

**Important**: When switching embedding providers, rebuild collections:
```bash
> --rag-build collection-name  # Required after provider change
```

## Exiting

Use `quit`, `exit`, `q`, or `Ctrl+C` to exit the application.

**Note**: Press `q` + `Enter` during AI responses to interrupt streaming, not to exit.

---

For issues or questions, please open an issue on the repository.