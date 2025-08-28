# Terminal AI Assistant

![Terminal AI Assistant Screenshot](screenshot.png)

A powerful terminal-based AI assistant that combines the best of conversational AI with advanced features like web search, RAG, youtube/article extraction, markdown rendering, multi-provider support and more.

Created from the desire to build a terminal alternative to [OpenWebUI](https://github.com/open-webui/open-webui).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
- [Content Input](#content-input)
- [Search & Research](#search--research)
- [Document Analysis (RAG)](#document-analysis-rag)
- [AI Customization (think "GPTs")](#ai-customization-think-gpts)
- [Conversation Management](#conversation-management)
- [Configuration](#configuration)
- [Shell Integration](#shell-integration)

## Features

- **Multi-Provider AI**: OpenAI, Google Gemini, and Ollama (local models)
- **Intelligent Web Search**: Real-time information via Tavily API or SearXNG with dynamic intent analysis
- **Deep Research Mode**: Autonomous research agent with intelligent termination
- **RAG System**: Query your documents with hybrid search and intelligent retrieval
- **Content Extraction**: YouTube transcripts, website content
- **Conversation Management**: Save, resume, and organize conversations
- **Instruction Templates**: Custom AI behaviors and skills
- **Clipboard Integration**: Use clipboard content as input
- **Privacy-First**: Local processing with Ollama for sensitive documents
- **Rich Commands**: Extensible system with tab completion

## Installation

### Prerequisites
- Python 3.10+
- [Conda](https://www.anaconda.com/docs/getting-started/) (Miniconda recommended)
- API keys (optional, but recommended):
  - **OpenAI API key** - For GPT models and OpenAI embeddings
  - **Google API key** - For Gemini models (free tier available)
  - **Tavily API key** - For web search functionality
- [Ollama](https://ollama.com/) (for local models)

### Setup

1. **Set up API keys securely**:
   
   Terminal AI uses **environment variables** for secure API key storage - keys are never stored in config files or source code.
   
   Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, or equivalent):
   ```bash
   export OPENAI_API_KEY="your-openai-key-here"
   export GOOGLE_API_KEY="your-google-key-here"
   export TAVILY_API_KEY="your-tavily-key-here"
   ```
   
   **Security benefits:**
   - Keys are kept out of version control
   - No risk of accidentally sharing keys in config files
   - Standard security practice

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
> Hello world

# Web search and analysis
> --search
> What's the latest news about AI?

# Analyze a YouTube video
> --youtube https://youtube.com/watch?v=example
> Summarize the key points

# Work with documents
> --file document.pdf
> What are the main conclusions?

# Switch to local model
> --model hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K_XL
> Tell me about quantum computing

# Exit the application
> quit
Confirm quitting (Y/n)?
```

## Core Commands

| Command | Description |
|---------|-------------|
| `--` | Show all available commands |
| `--model <name>` | Switch AI model (dynamically fetched from providers) |
| `--model-clear-cache` | Force refresh of available models |
| `--clear` | Clear conversation history |
| `--usage` | Show token usage and costs |

## Content Input

### File Processing

| Command | Description |
|---------|-------------|
| `--file <path>` | Load file contents (text, PDF, etc.) |
| `--folder <path>` | Load all supported files from directory (non-recursive) |
| `--folder-recursive <path>` | Load all supported files from directory and subdirectories |
| `--cb` | Use clipboard contents |
| `--cbl` | Copy latest AI reply to clipboard |

### URL Content Extraction

Extract content from websites/youtube with automatic paywall/access block bypass using multiple methods:

| Command | Description |
|---------|-------------|
| `--url <url>` | Extract website content |
| `--youtube <url>` | Extract website content |

**Bypass Methods:**
- Search engine bot headers (example: imitating Facebook crawler often allows access)
- Print versions
- AMP versions
- Archive.org (Wayback Machine)

### YouTube Integration

| Command | Description |
|---------|-------------|
| `--youtube <url>` | Extract video transcript |

Tip! `--url` will redirect to `--youtube` when required, so you really only need to use the `--url` command.

## Search & Deep Research

### Web Search

Intelligent web search with configurable search engines (Tavily or SearXNG) featuring LLM query optimization, generation, context awareness, and parallel processing for faster results.

**Basic Commands:**
| Command | Description |
|---------|-------------|
| `--search` | Toggle web search mode |
| `--search-auto` | AI decides per prompt whether to run a regular web search (not deep) |
| `--search-engine <engine>` | Switch search engine (tavily/searxng) for current session. Can be overriden in ~/.config/terminal-ai/config (as can any setting) |

**Search Engine Comparison:**

| Feature | Tavily | SearXNG |
|---------|--------|---------|
| **Setup Complexity** | Easy - just API key | Technical - requires self-hosted instance or public instance with JSON API enabled |
| **Cost** | Pay-per-search (~$0.005/search) | Free |
| **Privacy** | Commercial service | Fully private/self-hosted |
| **Content Quality** | AI-optimized results | Raw search engine results with content extraction |
| **Speed** | Fast | Slower (because web page content needs to be extracted for every result) |

**Which Should You Choose?**
- **Choose Tavily** if you want: Easy setup, don't mind costs, want AI-optimized results
- **Choose SearXNG** if you want: Complete privacy, free operation, full control over search sources
- **Both** options yield pretty much the same quality/end result because the search agent is optimized for both.

**Example Usage:**
```bash
# Enable web search
> --search

> What was the real secret behind the dharma initiative?
 • Generating search queries...
   • Dharma Initiative true purpose and secrets explained
   • Real secret behind the Dharma Initiative in Lost TV show
 • Search engine: Tavily
 • Searching (1/2): Dharma Initiative true purpose and secrets explained
   • 20 Years Later, the DHARMA Initiative Secret Finally Explained
   • DHARMA Initiative | Lostpedia - Fandom
   • Dharma Initiative - Wikipedia
 • Searching (2/2): Real secret behind the Dharma Initiative in Lost TV show
   • How the DHARMA Initiative Works | HowStuffWorks - Entertainment
   • The Others | Lostpedia | Fandom
 • Synthesizing 5 sources...
```

**Smart Features:**
- Conversation context awareness
- Current date awareness
- Parallel query execution to significantly reduce search response times
- Multi-query strategy (1-3 optimized searches per request)

### Auto Web Search (--search-auto)

- Lets the AI decide per prompt whether to run a regular web search.
- Uses recent conversation context for the decision and to optimize queries.
- Avoids unnecessary browsing for simple, timeless questions (e.g., 2 + 2).
- If SEARCH is chosen, it reuses the regular --search flow (not deep search).
- Transparency: prints “Auto-search decision: SEARCH” or “Auto-search decision: NO_SEARCH”.

Mutual exclusivity
- Enabling `--search-auto` disables `--search` and `--search-deep`.
- Enabling `--search` or `--search-deep` disables `--search-auto`.

Examples
```bash
> --search-auto
Auto web search enabled

> what is 2 + 2?
Auto-search decision: NO_SEARCH
4

> what happened in US politics over the past 24 hours?
Auto-search decision: SEARCH
 • Generating search queries...
 ...
```

**SearXNG Configuration:**

To use SearXNG, you need a running SearXNG instance with JSON output enabled. Configure your instance URL in `~/.config/terminal-ai/config`:

```bash
search_engine=searxng
searxng_base_url=http://your-searxng-instance:8080
```

**Important:** Your SearXNG instance must have JSON format enabled. Add this to your SearXNG `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Without JSON support, SearXNG integration will not work.

**SearXNG Content Extraction Settings** *(SearXNG only - ignored when using Tavily)*:

```bash
# Limit extracted content to first 2500 words per page (configurable word count to avoid exceeding context limits of whatever LLM your are using)
searxng_extract_full_content_truncate = 8000

# Cap each extraction method (Trafilatura, basic fetch, bypass attempts).
# 5s is typically fine on stable broadband. Consider 8–10s on slower or mobile networks,
# high-latency links, or when sites slow-walk bots.
extraction_method_timeout_seconds = 5
```

**How Content Extraction Works:**
- **SearXNG content extraction**: The app always downloads and extracts the entire content of each search result webpage, providing detailed information with paywall bypass capabilities. This makes SearXNG searches slower but much more comprehensive than basic snippets.
- **`searxng_extract_full_content_truncate`**: Sets the maximum number of words from each extracted webpage that will be included in the AI's context. This helps prevent exceeding the LLM's context window limit.

**Note:** Tavily handles content extraction automatically - these settings have no effect when `search_engine = tavily`.

### Deep Research Mode

The `--search-deep` command enables autonomous, intelligent web research that dynamically determines when sufficient information has been gathered to comprehensively answer complex queries, while still allowing user input to refine or conclude research.

| Command | Description |
|---------|-------------|
| `--search-deep` | Toggle autonomous deep search mode |

#### How Deep Research Works

Unlike basic search which performs a fixed number of searches, deep search uses an AI research agent:

1. **Dynamic Initial Research**: AI determines how many initial search queries are needed based on query complexity
2. **Autonomous Evaluation**: Analyzes gathered information and scores completeness (1-10 scale)
3. **Gap Analysis**: Identifies specific missing information or perspectives
4. **User Choice**: When below 10/10, offers user choice to continue or stop research
5. **Targeted Follow-up**: Generates precise searches to fill identified gaps
6. **Smart Termination**: Detects diminishing returns to prevent infinite search loops

#### Example Deep Research Session

```bash
User (samantha, markdown, gemini-2.5-flash):
> --search-deep
 • Deep search enabled - AI will autonomously determine research completeness

User (samantha, search-deep, markdown, gemini-2.5-flash):
> Write comprehensive report on current US climate policies
 • Deep Search Mode Activated
 • Search engine: Tavily
 • AI will autonomously determine when sufficient information has been gathered...
 • Initial Research Strategy:
   • 1. Current US federal climate policies legislation
   • 2. US climate regulations executive orders agencies
   • 3. Inflation Reduction Act climate provisions impact
   • 4. US clean energy and transportation climate policies
   • 5. US climate finance and investment initiatives
   • 6. US climate policy challenges and future outlook
 • Search 1: Current US federal climate policies legislation
   • Biden Sets Goal of Cutting Greenhouse Gases by More Than 60% - Newsweek
   • Climate policy outlook: 4 stories to know this week - GreenBiz
   • Florida braces for potential double-digit billion-dollar insurance losses after Hurricane Milton - The Daily Climate
   • Climate and environment updates: US generated record solar and wind energy - ABC News
 • Search 2: US climate regulations executive orders agencies
   • What to know about Trump's first executive actions on climate and environment - Idaho State Journal
   • EPA undertakes deregulatory actions, WOTUS revision - WaterWorld Magazine
   • Additional Recissions of Harmful Executive Orders and Actions - The White House
   • Navigating the Trump Administration’s Pause on IIJA and IRA Funding: Key Implications for Infrastructure Stakeholders - Crowell & Moring LLP
   • The Beginning of the End for the Climate-Industrial Complex - substack.com
 • Search 3: Inflation Reduction Act climate provisions impact
   • One Big Beautiful Bill – The Cost Of Climate Inaction - Forbes
   • If Trump Destroys Inflation Reduction Act, Economic Fallout May Come
   • Inflation Reduction Act Two Years Later: Clean Manufacturing ...
   • Project 2025 Offers A False Choice: Climate Action Vs. Economic ...
   • Death, Destruction—And Distraction: New Study On Media's Climate ...
 • Search 4: US clean energy and transportation climate policies
   • Will the success of Biden’s clean energy policies impede Trump’s agenda? - Utility Dive
   • California Democrats scale back climate goals amid cost-of-living backlash - The Daily Climate
 • Search 5: US climate finance and investment initiatives
   • The Sudden Shift in Big Banks' Stance on Fighting Climate Change
   • Bloom Energy (BE): Among the Best Climate Change Stocks to Buy ...
   • The $9 Trillion Climate Opportunity Hiding In Plain Sight - Forbes
   • Biggest Climate Fund Approves Record Allocations as US Withdraws
 • Search 6: US climate policy challenges and future outlook
   • What is the Future of U.S. Climate Policy? - Yahoo! Voices
   • Everyone Is Asking What The Future of U.S. Climate Policy Will Be - TIME
   • The Road to Belém: COP30 President on Trump, Trade, and What Comes Next - TIME
   • Investors bet on carbon removal technologies despite hurdles - The Daily Climate
 • Research Evaluation (after 6 searches):
   • Completeness: 7/10
   • Assessment: The gathered information provides a good overview of major US climate policy initiatives (like the IRA and Biden's NDC targets) and significant challenges,
     particularly the potential for policy shifts under a new administration. However, it suffers from a critical temporal ambiguity, presenting hypothetical future events
     (early 2025 under a Trump presidency) as current facts, which undermines its ability to comprehensively describe 'current' US climate policies as of the present moment
     (mid-2024). It also lacks granular detail on specific regulations and a broader perspective on state/local actions.
 • Decision: Research quality is good (7/10). Continue searching for higher completeness?
 • Gaps identified: Temporal Clarity: The information heavily features events and executive orders from early 2025 under Trump, assuming a specific outcome of the 2024 presidential election. A comprehensive report on 'current' policies needs to clearly distinguish between policies currently in effect under the Biden administration and potential
   or projected changes under a future administration. The current data blurs this distinction., Depth of Policy Detail: While major legislation (IRA, IIJA) and broad targets
   are mentioned, the report lacks specific details on the mechanisms, programs, and regulations within these policies. For example, specific EPA regulations for different
   sectors (e.g., power plants, vehicles, industry) are not detailed., Comprehensive Coverage of Current Administration's Policies: Before discussing potential rollbacks, a
   comprehensive report should thoroughly detail the existing climate policies, initiatives, and regulatory actions implemented by the Biden administration and its agencies
   (EPA, DOE, etc.) as they stand currently., Role of State and Local Policies: While California's efforts are briefly noted, a 'comprehensive report on US climate policies'
   should include a more detailed discussion of the significant role and variety of climate actions undertaken at the state and local levels, and how they interact with federal
   policy., Challenges and Effectiveness of Existing Policies: Beyond the future political challenges, a comprehensive report should also address the implementation challenges,
   economic hurdles, and effectiveness of the current climate policies in meeting their stated goals., International Context (beyond Paris Agreement): While the NDC is
   mentioned, a more comprehensive report could briefly touch upon other international climate diplomacy efforts or global pressures influencing US policy.
 • [C]ontinue deep search or [S]top and generate response
    Choice:
```

The user can choose to continue research or write a report based on the findings so far.

#### Deep Research vs Regular Search

| Feature | Regular Search | Deep Research |
|---------|---------------|-------------|
| **Search Strategy** | Fixed queries | Dynamic queries |
| **Evaluation** | None | AI evaluates completeness |
| **User Control** | None | Choice to continue/stop |
| **Termination** | After initial searches | When 10/10 or user stops |
| **Deduplication** | Basic | Advanced content deduplication |
| **Best For** | Quick facts | Complex research |
| **Sources** | 3-6 sources | many more sources (dynamic) |

## Document Analysis (RAG)

RAG allows you to query your own documents with intelligent, temporal retrieval.

### Setup

```bash
# Create and add documents to RAG directory
mkdir -p ~/path/to/terminal-ai/rag/personal
mkdir -p ~/path/to/terminal-ai/rag/work
mkdir -p ~/path/to/terminal-ai/rag/whatever

# Activate collection (builds automatically)
# See settings_manager.py for default RAG embedding models (cloud and local), override in ~/.terminal-ai/config
> --rag personal
> Detail my latest visit to the doctor
```

### Commands

| Command | Description |
|---------|-------------|
| `--rag [collection]` | Activate specific collection |
| `--rag-rebuild <collection>` | Rebuild embeddings index (smart rebuild by default) |
| `--rag-rebuild <collection> --force-full` | Force complete rebuild from scratch |
| `--rag-show <filename>` | View relevant chunks from file given as source post RAG query |
| `--rag-status` | Show RAG configuration |
| `--rag-test` | Test embedding provider connection |
| `--rag-deactivate` | Deactivate RAG |

### Smart Rebuild

RAG collections use smart rebuild by default, which only processes changed files instead of rebuilding everything from scratch. This makes rebuilding even large RAG collections very fast (after initial build).

Use `--rag-rebuild collection --force-full` to force a complete rebuild if desired.

### Embedding Providers

- **OpenAI**: High quality, cloud-based (requires API key)
- **Ollama**: Local, private, free (e.g. `ollama pull snowflake-arctic-embed2:latest`) (also good quality)

## AI Customization (think "GPTs")

### Instruction Templates

Create custom AI behaviors in `instructions/`:

| Command | Description |
|---------|-------------|
| `--instructions <file>` | Apply instruction template |

See default in `settings_manager.py`. Override in `~/.config/terminal-ai/config`.

**Example Usage:**
```bash
> --instructions summary.md
> --youtube https://youtube.com/watch?v=example
> Summarize this video
```

### Mode Toggles

| Command | Description |
|---------|-------------|
| `--markdown` | Toggle markdown rendering |
| `--nothink` | Disable thinking on Ollama models |

## Conversation Management

Terminal-AI uses a JSON logging system that provides reliable conversation storage and optional markdown export.

### Logging System Overview

**File Structure:**
- `logs/` - Contains JSON conversation files (authoritative records)
- `logs/export/` - Contains exported markdown files (optional export)

**How it Works:**
- All conversations are automatically saved as JSON files in `logs/{instruction-set}/`
    - Use `--incognito` to disable logging
- JSON files contain complete conversation history including system messages
- Markdown files are only created when explicitly exported using `--export-markdown`
- JSON files are the single source of truth for all conversation data

### Commands

| Command | Description |
|---------|-------------|
| `--log <filename>` | Resume previous conversation from JSON file |
| `--logmv [title]` | Rename current conversation |
| `--logrm` | Delete current conversation |
| `--export-markdown [title]` | Export current conversation to markdown ([title] is optional) |
| `--incognito` | Toggle private mode (no logging) |

### Usage Examples

```bash
# Resume a conversation (works with or without file extension)
# Will display a list of available conversations sorted by date
--log my-conversation

# Export current conversation to markdown
--export-markdown
--export-markdown custom-title

# Rename current conversation
--logmv project-discussion
--logmv  # AI will suggest title based on conversation content

# Delete current conversation
--logrm
```

### File Organization

Conversations are organized by instruction set:
```
logs/
├── samantha/
│   ├── conversation_20250818-193240.json
│   ├── project-discussion_20250817-133240.json
│   └── export/
│       ├── conversation_20250818-193240.md
│       └── project-discussion_20250817-133240.md
└── rewrite/
    ├── code-review_20250810-203240.json
    └── export/
        └── code-review_20250810-203240.md
```

## Configuration

You can override any setting from `settings_manager.py` by creating `~/.config/terminal-ai/config`. See `config/config.example`.

Settings follow a three-tier priority system: `--input` commands override config file settings, which override defaults from `settings_manager.py`.

### Configuration Features

- Simple `setting = value` format
- Automatic type conversion (booleans, numbers, strings)
- Graceful fallback to defaults for missing settings
- Invalid setting warnings

### Key Settings

Key settings in `src/settings_manager.py`:

```python
# Default model and provider settings
self.default_model = "gpt-5-mini"
self.embedding_provider = "ollama"  # or "openai"
self.ollama_embedding_model = "snowflake-arctic-embed2:latest"
self.openai_embedding_model = "text-embedding-3-small"
self.chunk_size = 1000
self.chunk_overlap = 200
self.rag_batch_size = 16

# MORE...
# MUCH MORE...
# see settings_manager.py for all settings
```

## Shell Integration

Terminal AI becomes incredibly powerful when integrated into your shell workflow. Instead of starting the app and typing the same thing over and over for common tasks, you can create short, memorable aliases that handle complex AI operations.

### Helper Script Pattern

The most effective approach is to create a helper script with AI workflow functions, then source it in your shell profile.

Example; perform a web search for "latest AI developments"
```bash
sw "latest AI developments"  # web search + AI analysis
```

### Getting Started

1. **Copy the example script**: Use `shell.example.sh` as your starting point
2. **Customize models**: Edit the `AI_MODEL` and `AI_MODEL_LOCAL` variables
3. **Update path**: Change `TERMINAL_AI_PATH=~/Sync/scripts/terminal-ai` to your actual terminal-ai path
4. **Source in your shell**:
```bash
# Add to ~/.bashrc or ~/.zshrc
source ~/path/to/shell.example.sh
```
5. **Get help**: Run `ai` to see all available shortcuts and you will see (example below shows what's included in `shell.example.sh`):

```bash
Terminal AI Helper Functions:
=============================
cb:        Analyze clipboard content
cbl:       Local (Ollama): Analyze clipboard content
cli:       CLI/system administration expert mode
clil:      Local (Ollama): CLI/system administration expert mode
rewrite:   Rewrite/improve text from clipboard
rewritel:  Local (Ollama): Rewrite/improve text from clipboard
s:         Basic AI conversation with cloud model
sl:        Local (Ollama): Basic AI conversation
su:        Quick URL summary (3 key bullet points)
sul:       Local (Ollama): Quick URL summary
sw:        Web search + AI analysis - great for current events and research
swl:       Local (Ollama): Web search + AI analysis
swd:       Deep web search + comprehensive AI analysis - for complex research
swdl:      Local (Ollama): Deep web search + comprehensive AI analysis
u:         Summarize URL content (paste URL or use clipboard)
ul:        Local (Ollama): Summarize URL content
```

### Example Workflows

The example script includes these common patterns:

**Cloud & Local Model Pairs**: Every function has both cloud and local versions. Local versions (ending with 'l') use your local model and show "Local (Ollama):" in help text.

- **`s`** / **`sl`** - Basic conversation
    - Example: `s "explain quantum computing"`
    - Example: `sl "explain quantum computing"`
- **`sw`** / **`swl`** - Web search + AI analysis
    - Example: `sw "latest developments in AI"`
    - Example: `swl "latest developments in AI"`
- **`swd`** / **`swdl`** - Deep web search + comprehensive AI analysis
    - Example: `swd "Write report on current climate change policies in the US"`
    - ...
- **`cb`** / **`cbl`** - Analyze clipboard content
    - Example: `cb "explain this code"`
    - ...
- **`u`** / **`ul`** - Process URL content (automatic clipboard detection)
    - Example: `u "https://ubuntu.com/download/desktop"` (adds page content to conversation)
    - Example: `u "https://ubuntu.com/download/desktop" "what's the latest version?"`
    - ...
- **`su`** / **`sul`** - Quick URL summary (3 key bullet points)
    - Example: `su "https://www.youtube.com/watch?v=6mp_CGzx6p4"`
    - Example: `su` (automatically uses URL from clipboard)
    - ...
- **`cli`** / **`clil`** - Return specific CLI commands
    - Example: `cli "how to restart apache"`
    - ...
- **`rewrite`** / **`rewritel`** - Rewrite/improve clipboard text
    - Example: `rewrite` (improves text from clipboard)
    - ...

You can easily rename functions, add new ones, or do whatever you want.

### Advanced Usage

The helper script pattern scales to any workflow:
- **Model switching**: Every function has cloud/local versions (e.g., `s`/`sl`, `cb`/`cbl`)
- **Clipboard automation**: URL functions automatically detect clipboard URLs when no URL provided
- **Specialized modes**: Custom instructions for different domains (CLI, rewriting)

### Zsh Tip

Add `setopt nonomatch` to your `~/.zshrc` to enable unquoted arguments:
```bash
s hello world        # Instead of s "hello world"
sw latest AI news    # Instead of sw "latest AI news"
```

See `shell.example.sh` for detailed implementation examples.

### Exit Methods
- Type `quit`, `q`, `:q`, `:wq` in prompt to exit application
- Press `q` + `Enter` during AI responses to interrupt (not exit)

---

For support or questions, please open an issue on the repository.