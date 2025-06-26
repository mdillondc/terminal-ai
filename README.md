# Terminal AI Assistant

![Terminal AI Assistant Screenshot](screenshot.png)

A powerful terminal-based AI assistant that combines the best of conversational AI with advanced features like RAG (Retrieval-Augmented Generation), web search, command execution, and multi-provider support.

Created from the desire to build a terminal alternative to [OpenWebUI](https://github.com/open-webui/open-webui).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Configurations](#configuration)
- [Usage](#usage)
  - [Core Commands](#core-commands)
  - [Content Processing](#content-processing)
  - [RAG (Document Analysis)](#rag-document-analysis)
  - [Web Search](#web-search)
  - [Conversation Management](#conversation-management)
  - [Text-to-Speech](#text-to-speech)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Shell Integration](#shell-integration)

## Features

### Core Capabilities
- **Multi-Provider AI**: OpenAI, Google Gemini, Anthropic, and Ollama (local models)
- **Intelligent Web Search**: Real-time information via Tavily API with dynamic intent analysis (temporal, factual, controversial, etc.)
- **RAG System**: Query your documents with hybrid search and intelligent retrieval
- **Content Extraction**: YouTube transcripts, website content with paywall bypass

### Advanced Features  
- **Conversation Management**: Save, resume, and organize conversations
- **Text-to-Speech**: Natural speech synthesis with multiple voices
- **Instruction Templates**: Custom AI behaviors and skills
- **Clipboard Integration**: Use clipboard content as input
- **Privacy-First**: Local processing with Ollama for sensitive documents
- **Rich Commands**: Extensible system with tab completion

## Installation

### Prerequisites
- Python 3.10+
- [Conda](https://www.anaconda.com/docs/getting-started/) (Miniconda recommended)
- API keys (optional):
  - OpenAI API key
  - Google API key  
  - Anthropic API key
  - Tavily API key (for web search)
- [Ollama](https://ollama.com/) (for local models)

### Setup

1. **Set up API keys** (add to your shell profile):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GOOGLE_API_KEY="your-google-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export TAVILY_API_KEY="your-tavily-key"
   ```

2. **Install Terminal AI**:
   ```bash
   git clone https://github.com/mdillondc/terminal-ai
   cd terminal-ai
   conda create -n terminal-ai python=3.12 -y
   conda activate terminal-ai
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python src/main.py
   ```

## Quick Start

```bash
# Start the application
python src/main.py

# Basic usage
> --model gpt-4.1
> Hello! How can you help me today?

# Enable web search
> --search
> What's the latest news about AI?

# Analyze a YouTube video
> --youtube https://youtube.com/watch?v=example
> Summarize the key points

# Work with documents
> --file document.pdf
> What are the main conclusions?

# Enable text-to-speech
> --tts
> Tell me a joke
# The joke will be spoken aloud

# Exit the application
> quit
```

### Configuration

You can override any setting from `settings_manager.py` by creating `~/.config/terminal-ai/config`. See `config/config.example`.

Settings follow a three-tier priority system: `--input` commands override config file settings, which override defaults from `settings_manager.py`.

**Features:**
- Simple `setting = value` format
- Comments supported with `#`
- Automatic type conversion (booleans, numbers, strings)
- Graceful fallback to defaults for missing settings
- Invalid setting warnings

## Usage

### Core Commands

| Command | Description |
|---------|-------------|
| `--` | Show all available commands |
| `--model <name>` | Switch AI model (dynamically fetched from providers) |
| `--model-clear-cache` | Force refresh of available models |
| `--clear` | Clear conversation history |
| `--usage` | Show token usage and costs |

### Content Processing

| Command | Description |
|---------|-------------|
| `--file <path>` | Load file contents (text, PDF, images, etc.) |
| `--url <url>` | Extract website content with paywall bypass |
| `--youtube <url>` | Extract video transcript |
| `--cb` | Use clipboard contents |
| `--cbl` | Copy latest AI reply to clipboard |
| `--instructions <file>` | Apply instruction template |

### RAG (Document Analysis)

RAG allows you to query your own documents with intelligent retrieval.

**Setup:**
```bash
# Create document collection
mkdir -p rag/my-docs
# Add your documents to the directory

# Activate collection (builds automatically)
> --rag my-docs
> Detail my latest visit to the doctor
```

**Commands:**
| Command | Description |
|---------|-------------|
| `--rag [collection]` | Toggle RAG or activate specific collection |
| `--rag-rebuild <collection>` | Force rebuild embeddings index |
| `--rag-show <filename>` | View relevant chunks from file |
| `--rag-status` | Show RAG configuration |
| `--rag-test` | Test embedding provider connection |

**Embedding Providers:**
- **OpenAI**: High quality, cloud-based (requires API key)
- **Ollama**: Local, private, free (install: `ollama pull snowflake-arctic-embed2:latest`)

### Web Search

Intelligent web search with dynamic intent analysis (temporal, factual, controversial, etc).

```bash
# Enable web search
> --search

> list 5 of trumps crimes
- Analyzing search intent...
- Intent analysis: controversial query (confidence: 0.95)
- Generating optimal search query...
- Generated search queries: Donald Trump criminal charges and allegations 2025, List of crimes and legal cases involving Donald Trump 2025, Recent updates on Donald Trump criminal investigations 2025
- Searching: Donald Trump criminal charges and allegations 2025 (depth: advanced)
- Searching: List of crimes and legal cases involving Donald Trump 2025 (depth: advanced)
- Search completed. Analyzing results...
```

**Smart Features:**
- Dynamic intent detection via LLM analysis
- Automatic fact-checking for controversial topics
- Conversation context awareness
- Current date awareness in queries
- Multi-query strategy (1-3 optimized searches per request)

### Conversation Management

| Command | Description |
|---------|-------------|
| `--log <filename>` | Resume previous conversation |
| `--logmv [title]` | Rename current conversation |
| `--logrm` | Delete current conversation |
| `--incognito` | Toggle private mode (no logging) |

### Text-to-Speech

| Command | Description |
|---------|-------------|
| `--tts` | Toggle text-to-speech |
| `--tts-model <model>` | Change TTS model |
| `--tts-voice <voice>` | Select voice |
| `--tts-save-as-mp3` | Save responses as MP3 files |

## Advanced Features

### Command Logging
All commands executed with `--` prefix and their output are automatically logged to conversation files (`.md` and `.json`), providing complete audit trails of your AI interactions. This includes status messages, progress updates, and command results for easy debugging and workflow documentation.

### URL Content Extraction
Automatic paywall/access block bypass using multiple methods:
- Search engine bot headers
- Print versions
- AMP versions
- Archive.org (Wayback Machine)

### Instruction Templates
Create custom AI behaviors/prompts in `instructions/`:
```bash
> --instructions summary.md
> --youtube https://youtube.com/watch?v=example
> Summarize this video
```

Find instruction inspiration at [fabric/patterns](https://github.com/danielmiessler/fabric/tree/main/patterns).

### Mode Toggles
| Command | Description |
|---------|-------------|
| `--search` | Toggle web search mode |
| `--markdown` | [EXPERIMENTAL] Toggle markdown rendering for AI responses |
| `--scroll` | Toggle scroll navigation (hotkey F8). Use j/k to scroll, gg for top, G for bottom |
| `--nothink` | Disable thinking on Ollama models |
| `--incognito` | Toggle private mode (no logs) |

## Configuration

Key settings in `src/settings_manager.py`:

```python
# Default model and provider settings
self.default_model = "gpt-4.1"
self.embedding_provider = "ollama"  # or "openai"
self.ollama_embedding_model = "snowflake-arctic-embed2:latest"
self.openai_embedding_model = "text-embedding-3-small"

# RAG configuration
self.chunk_size = 1000
self.chunk_overlap = 200
self.top_k_results = 5

# TTS settings
self.tts_voice = "nova"
self.tts_model = "tts-1"

# More
# see settings_manager.py
```

## Shell Integration

Create powerful AI workflows with shell aliases:

```bash
# Basic alias
alias ai='python ~/terminal-ai/src/main.py'

# Article summarizer
summarize() {
    python ~/terminal-ai/src/main.py \
        --input "--model gpt-4.1 --instructions summary.md --url $1" \
        --input "summarize this article"
}

# YouTube analyzer  
youtube() {
    python ~/terminal-ai/src/main.py \
        --input "--model gpt-4.1 --instructions summary.md --youtube $1" \
        --input "summarize this video"
}

# Usage examples:
# summarize "https://example.com/article"
# youtube "https://youtube.com/watch?v=example"
```

## Troubleshooting

### Common Issues

**Model not found:**
```bash
> --model-clear-cache
> --model gpt-4.1
```

**RAG not working:**
```bash
> --rag-test
> --rag-rebuild collection-name
```

**Ollama connection:**
```bash
# Check if Ollama is running
ollama list
# Start Ollama service if needed
ollama serve
```

### Exit Methods
- `quit`, `q`, `:q`, `:wq`, or `Ctrl+C`
- Press `q` + `Enter` during AI responses to interrupt (not exit)

---

**Note**: This project is actively maintained and new features are added regularly based on user feedback and personal workflow needs.

For support or questions, please open an issue on the repository.