# Terminal AI Assistant

![Terminal AI Assistant Screenshot](screenshot.png)

A terminal AI assistant with advanced features including RAG, web search, command execution, TTS, multiple providers (including Ollama) and more.

As a long-time user of [OpenWebUI](https://github.com/open-webui/open-webui) and [open-interpreter](https://github.com/OpenInterpreter/open-interpreter), I found myself wishing that I could bring them both into a single, cohesive terminal application. To achieve this, I created Terminal AI.

I use this program every day on Linux. It should work on macOS as well. If you use Windows, I hope you seek mental health help from a trained professional.

## Features

- **Multi-Provider AI**: OpenAI, Google Gemini, and Ollama (local models)
- **RAG System**: Query your own documents with intelligent retrieval and hybrid search (analyzes query for tone: temporal, factual, analytical, etc)
- **Web Search**: Real-time information via Tavily API (optimized for AI)
- **Command Execution**: Execute system commands with AI assistance and permission controls (think [open-interpreter](https://github.com/OpenInterpreter/open-interpreter))
- **YouTube Integration**: Extract and analyze video transcripts
- **URL Content Extraction**: Extract and analyze content from any website with intelligent paywall bypass
- **Conversation Management**: Save, resume, and organize conversations
- **Text-to-Speech**: Convert responses to natural speech
- **Rich Commands**: Extensible command system with tab completion
- **Clipboard Integration**: Use clipboard content as input

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API key (required, optional if you want to use Ollama models)
- Tavily API key (optional, for web search)
- Google API key (optional, for Gemini models)
- Ollama (optional, for local models and private RAG)

### Setup

0. **Install Conda**

Conda let's you create isolated environments for Python projects. Get it [here](https://www.anaconda.com/docs/getting-started/). It's free. Miniconda is recommended.

1. **Set API keys**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export TAVILY_API_KEY="your-tavily-key"
   export GOOGLE_API_KEY="your-google-key"
   ```
      
2. **Clone and set up environment**
   ```bash
   git clone https://github.com/mdillondc/terminal-ai
   cd terminal-ai
   conda create -n terminal-ai python=3.12 -y
   conda activate terminal-ai
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
> --model gpt-4.1

# Enable web search
> --search

# Enable command execution (think open-interpreter)
> --execute

# Apply instruction (think gpts) templates
> --instructions summary.md

# Enable text-to-speech
> --tts
> --tts-voice nova
> --tts-save-as-mp3 # Save audio files locally

# Analyze YouTube video
> --youtube https://youtube.com/watch?v=example
> Summarize video

# Enable command execution (think open-interpreter)
> --execute

# Extract content from any website
> --url https://example.com/article

# Load file contents into conversation context
> --file path/to/document.pdf

# Use clipboard content
> --cb
```

### Web Search

Terminal AI includes intelligent web search with dynamic intent analysis - no hardcoded patterns, fully adaptive to any topic.

```bash
# Enable web search
> --search

# Search automatically analyzes your query for:
# - Intent type (factual, controversial, recent events, etc.)
# - Search depth (basic vs advanced)
# - Freshness requirements
# - Verification needs

# Example searches:
> How many people were affected by the earthquake in Turkey?
# - Analyzes as: numerical query requiring verification
# - Uses: advanced search depth
# - Adds: verification searches for accurate casualty figures
# - Results: Multiple sources as numbers get updated over time

> What's the capital of France?
# - Analyzes as: simple factual query  
# - Uses: basic search depth
# - Results: Quick, authoritative answer

> Latest AI breakthrough this week?
# - Analyzes as: recent events query
# - Uses: advanced search with freshness filtering (7 days)
# - Results: Most current information
```

**Smart Features:**
- **Dynamic Intent Detection**: LLM analyzes each query contextually (no hardcoded patterns)
- **Verification Searches**: Automatically fact-checks controversial topics
- **Conversation Context**: Understands references like "his parade" from previous discussion
- **Current Date Awareness**: Always uses correct year/date in search queries
- **Multi-Query Strategy**: Generates 1-3 optimized search queries per request

### RAG (Document Analysis)

```bash
# Create document collection (supports various formats, see `rag_config.py`)
mkdir -p rag/my-docs
mkdir -p rag/my-docs/api # Subdirectories supported
cp hello-world.md rag/my-docs/
cp api-docs/*.md rag/my-docs/api/

# Build and activate collection
> --rag-build my-docs
> --rag my-docs # enables RAG collection (will automatically rebuild collection when changes are detected)

# Query your documents
> What are the main topics in my documents?
> --rag-show filename.txt # View specific file chunks
> --rag-status            # Check RAG configuration
> --rag-test              # Test RAG connection
```

### Embedding Providers

**OpenAI** (cloud-based):
- High quality, requires API key
- No privacy

**Ollama** (local, private):
- Free, runs locally, works offline
- Install: `curl -fsSL https://ollama.com/install.sh | sh`
- Setup (example embedding model): `ollama pull snowflake-arctic-embed2:latest`

Configure models/providers in `src/settings_manager.py`:
```python
self.embedding_provider = "ollama"  # or "openai"
```

### Conversation Management

```bash
# Resume previous conversation
# AI will suggest log names automatically, so you really don't have to about this
> --log name-of-conversation.md

# Rename current conversation with AI suggested title (if you're unhappy about the first suggestion)
> --logmv

# Rename current conversation with your own title
> --logmv project-planning-discussion
> --logmv "Title with spaces"

# Private mode (no logs)
> --incognito
```

### URL Content Extraction & Paywall Handling

```bash
# Extract content from any website
> --url https://nytimes.com/some-article
 - Extracting content from URL...
 - Paywall detected, trying alternative methods...
 - Paywall bypassed using Archive.org
 - Content added to conversation context.

# YouTube URLs automatically redirect to --youtube command
> --url https://youtube.com/watch?v=example
 - YouTube URL detected - redirecting to --youtube command...

# Works with most urls
> --url https://theguardian.com/article
> --url https://medium.com/@author/story
```

**Paywall Bypass Methods** (attempted automatically):
- **Archive.org (Wayback Machine)** - Often most effective for older articles
- **Search Engine Bot Headers** - Many sites show full content to search engines
- **Print Versions** - URLs like `?print=1` often bypass paywalls
- **AMP Versions** - Google AMP pages sometimes bypass restrictions

**Note**: Paywall bypass success varies by site and article age. The feature will transparently inform you whether bypass attempts succeeded or failed.

### Getting Help

```bash
> -- # Show all available commands
```

## Configuration

### Instruction Templates

Create custom markdown instruction files in `instructions/`:

A good resource for instructions can be found at [fabric/patterns](https://github.com/danielmiessler/fabric/tree/main/patterns).

### Settings

Key settings in `src/settings_manager.py`:
- Default model selection and API configuration
- RAG parameters (chunk size, top-k results, embedding provider)
   - Adjust based on your embedding model's capabilities
- Logging parameters (log directory, log level)
- TTS voice and model settings
- Much more. See `src/settings_manager.py` for details

### Audio Files

When `--tts-save-as-mp3` is enabled, audio files are saved as:

```
tts_{timestamp}_{text_preview}.mp3
```

## File Structure

```
terminal-ai/
├── src/              # Application source code
├── instructions/     # Instruction templates
├── rag/              # Document collections
├── logs/             # Conversation logs
├── cache/            # Model and completion cache
└── requirements.txt  # Python dependencies
```

## Dependencies

See requirements.txt

## API Costs & Privacy

- **OpenAI**: Pay per token
- **Tavily**: Free tier available, paid plans for higher usage (generous free-tier)
- **Google**: Free tier available
- **Ollama**: Free (runs locally, fully private)

**Privacy Note**: Use Ollama for sensitive documents/RAG - data never leaves your machine.

## Quick Start Example

All command examples with --input can of course be used within the actual app.

```bash
python src/main.py --input "Hello Samantha!"

# You can add any available commands to the --input command, e.g.:
python src/main.py --input "--model qwen3:14b-q8_0 --instructions summary.md --youtube https://youtube.com/some-url" --input "summarize the video"

# Batch processing with multiple --input arguments (command-chaining)
python src/main.py --input "--model gpt-4.1-mini --instructions summary.md --youtube https://youtube.com/watch?v=example" --input "summarize the key points from this video"
# First input: configures model, applies instructions, extracts YouTube transcript
# Second input: asks AI to summarize the video content
# Then continues to interactive mode for follow-up questions

# Process local files with batch commands
python src/main.py --input "--file report.pdf --instructions summary.md" --input "extract key findings"

# Enable features and start chatting
> --tts --search
# App will tell you which features you've enabled
# User (tts, search):
> What's the latest news about AI?

# Work with documents (private with Ollama)
> --rag-build my-docs
> --rag my-docs
> Summarize the key findings from my research papers

# If you add a new document or change the content of an existing document
# in a collection, the RAG collection will automatically rebuild the next
# time you enable the RAG collection in question

# Configure for privacy in src/settings_manager.py:
# self.embedding_provider = "ollama"
# self.ollama_embedding_model = "snowflake-arctic-embed2:latest"
```

## Shell Aliases & Command Chaining

Create powerful AI aliases/workflows in your `~/.bashrc` or `~/.zshrc`:

```bash
# Basic alias
alias ai='python ~/terminal-ai/src/main.py --input "--model gpt-4.1"'

# Article summarizer
summarize() {
    if [ -z "$1" ]; then
        echo "Usage: youtube <url>"
        return 1
    fi

    conda activate terminal-ai && python ~/path/to/terminal-ai/src/main.py --input "--model gpt-4.1-mini --instructions summary.md --url $*" --input "summarize"
}
# First the app will set desired model, apply instructions, extract content from url, then summarize content from url
# Usage: analyze "https://example.com/article"

youtube() {
    if [ -z "$1" ]; then
        echo "Usage: youtube <youtube_url>"
        return 1
    fi

    conda activate terminal-ai && python ~/Sync/scripts/terminal-ai/src/main.py --input "--model gpt-4.1-mini --instructions summary.md --youtube $*" --input "Summarize video"
}
# First the app will set desired model, apply instructions, extract the youtube transcript, then summarize content from url
# Usage: youtube "https://www.youtube.com/watch?v=9cgbsavrFpA"
```

## Exiting

Use `quit`, `q`, `:q`, `:wq`, or `Ctrl+C` to exit the application.

**Note**: Press `q` + `Enter` during AI responses to interrupt streaming, not to exit.

## Todo

Honestly, I just keep adding features whenever I miss something.

Suggestions are welcome, please open an [issue](https://github.com/mdillondc/terminal-ai/issues).

* [ ] Maybe create a pip package for easy installation?
* [ ] Add lots of helpful `instructions/*.md`

---

For issues or questions, please open an issue on the repository.