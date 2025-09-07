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
- [Experimental: Image Generation & Editing](#experimental-image-generation--editing)

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

## Experimental: Image Generation & Editing

Create and edit images (experimental). Files are saved under `images/` as `model_YYYYMMDD-HHMMSS.png`. Two modes:

- Generate mode (toggle):
  - `--image-generate`; plain text prompts create new images
  - One‑shot: `--image-generate a german shepherd is sleeping on a couch`

- Edit mode (persistent, iterative):
  - `--image-edit images/your.png`
    `create 3d mesh around person in image`
  - `--image-edit` (no path) uses the last generated/edited image
      - `add a plush toy next to the dogs head`
  - Each edit builds on the latest result and saves a new file
```

## Content Input

### File Processing

| Command | Description |
|---------|-------------|
| `--file <path>` | Load file contents (text, PDF, images, etc.) |
| `--folder <path>` | Load all supported files from directory (non-recursive) |
| `--folder-recursive <path>` | Load all supported files from directory and subdirectories |
| `--cb` | Use clipboard contents |
| `--cbl` | Copy latest AI reply to clipboard |

### Image & Vision Support

- Image files with .jpg, .jpeg, .png are supported with the same commands you use for documents.
- `--file`:
  - When you pass an image, it’s analyzed using your configured vision model
  - The analysis (detailed description and extracted text) is added to the conversation context like document content.
- `--folder` and `--folder-recursive`:
  - These commands also include image files (.jpg, .jpeg, .png) in addition to regular supported file types.
  - If more than N images are found, you’ll be asked: "More than 5 (147) images found, do you want to include them them? (Y/n)"
- PDFs:
  - Image-only (scanned) pages in PDFs are automatically processed with the vision model during `--file` / `--folder` / `--folder-recursive` when no text is extractable.
  - Uses `cloud_vision_model` or `ollama_vision_model` based on your active chat provider.
- Configuration:
  - `pdf_vision_max_pages` (0 disables)
  - `pdf_vision_dpi`

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
- Transparency: prints "search-auto decision: SEARCH" or "search-auto decision: NO_SEARCH".

Mutual exclusivity
- The four modes are mutually exclusive: `--search`, `--search-auto`, `--search-deep`, `--search-deep-auto`.
- Enabling any one disables the others. `--search-engine` is independent.

Examples
```bash
> --search-auto
search-auto enabled

> what is 2 + 2?
search-auto decision: NO_SEARCH
4

> what happened in US politics over the past 24 hours?
search-auto decision: SEARCH
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

The `--search-deep` command enables autonomous, intelligent web research that dynamically determines when sufficient information has been gathered to comprehensively answer complex queries, while still allowing user input to refine or conclude research. The `--search-deep-auto` command lets the AI decide per prompt whether to run deep research (or not), printing "search-deep-auto decision: SEARCH" or "search-deep-auto decision: NO_SEARCH" before proceeding.

| Command | Description |
|---------|-------------|
| `--search-deep` | Toggle autonomous deep search mode |
| `--search-deep-auto` | Toggle automatic deep research mode. The AI evaluates each prompt and decides whether to run autonomous deep research (not regular search). |

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

RAG lets you query your own documents with hybrid retrieval:
- Semantic search (vector embeddings)
- Optional keyword/BM25 signals for result diversity and recency boosting

Ingestion flow: files are discovered in `rag/<collection>/` → chunked → embedded → stored → searched at query time. The active embedding provider (OpenAI vs Ollama) is selected automatically based on your current chat model/provider to ensure privacy.

Note: PDFs with scanned/image-only pages are automatically processed with the provider-specific vision model during ingestion and respect `pdf_vision_max_pages` (0 disables) and `pdf_vision_dpi`.

### Provider-Specific Indexes (Profiles)

Each collection maintains a separate index per embedding provider/model (“profile”). This avoids rebuild churn when you switch providers. The vectorstore layout is:

```
rag/vectorstore/openai/text-embedding-3-large/<collection>_index.parquet
rag/vectorstore/openai/text-embedding-3-large/<collection>_meta.parquet

rag/vectorstore/ollama/snowflake-arctic-embed2-latest/<collection>_index.parquet
rag/vectorstore/ollama/snowflake-arctic-embed2-latest/<collection>_meta.parquet
```

- The provider folder reflects the embedding provider: `openai` or `ollama`
- The model folder is the embedding model name, sanitized (non [A-Za-z0-9_-] replaced with `-`)
- Switching chat models switches the embedding provider and RAG automatically uses the matching profile. If a profile hasn’t been built yet, the first activation will build it. Subsequent runs use smart rebuild within that profile.

### Setup

```bash
# Create and add documents to RAG directory
mkdir -p ~/path/to/terminal-ai/rag/personal
mkdir -p ~/path/to/terminal-ai/rag/work
mkdir -p ~/path/to/terminal-ai/rag/whatever

# Activate collection (auto-selects provider/model profile and builds if needed)
# See settings_manager.py for default embedding models (cloud and local), override in ~/.config/terminal-ai/config
> --rag personal
> Detail my latest visit to the doctor
```

### Commands

| Command | Description |
|---------|-------------|
| `--rag [collection]` | Activate collection (uses the active provider/model profile) |
| `--rag-rebuild <collection>` | Rebuild the active profile’s index (smart rebuild by default) |
| `--rag-rebuild <collection> --force-full` | Force a full rebuild for the active profile |
| `--rag-show <filename>` | Show relevant chunks from a specific source file (active profile) |
| `--rag-status` | Show RAG status including provider, model, and embedding dimensions |
| `--rag-test` | Test connectivity for the current embedding provider |
| `--rag-deactivate` | Deactivate RAG |

### Smart Rebuild

Smart rebuild runs within the active provider/model profile and only re-embeds changed files. This makes iterative updates fast even for large collections.

- First build happens once per profile (e.g., once for OpenAI, once for Ollama)
- Switching providers does not trigger a rebuild unless the target profile has never been built

Use `--rag-rebuild <collection> --force-full` to force a complete rebuild for the active profile.

### Status & Configuration

`--rag-status` includes:
- `embedding_provider` (e.g., `openai`, `ollama`)
- `embedding_model` (e.g., `text-embedding-3-large`, `snowflake-arctic-embed2:latest`)
- `embedding_dimensions` (e.g., 3072, 1024)
- Chunking and retrieval settings (size/overlap/top_k)

Defaults live in `settings_manager.py`:
- `cloud_embedding_model` for OpenAI
- `ollama_embedding_model` for Ollama
Override in `~/.config/terminal-ai/config` if desired.

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
- When resuming a conversation with `--log`, Terminal‑AI maintains a single system timeline entry to help models reason about elapsed time:
  - An initial system date anchor is written once on new conversations: `Info:System-Date: YYYY-MM-DD`
  - On resume, the app consolidates to one up‑to‑date timeline line:  
    `Info:System-Timeline: start=YYYY-MM-DD; last=YYYY-MM-DD; current=YYYY-MM-DD; days_elapsed=N; days_inclusive=M; active_days=[YYYY-MM-DD,...]`
  - This feature can be disabled via `system_timeline_enabled = False` in your config

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
self.model = "gemini-2.5-flash"
self.cloud_embedding_model = "text-embedding-3-large"
self.ollama_embedding_model = "snowflake-arctic-embed2:latest"
self.cloud_vision_model = "gpt-5-mini"
self.ollama_vision_model = "qwen2.5vl:7b"
self.pdf_vision_max_pages = 999
self.pdf_vision_dpi = 180
self.gpt5_display_full_reasoning = False  # If True, GPT‑5 streams full reasoning; if False, show a generic "Working" status until first visible output
self.rag_chunk_size = 400
self.rag_chunk_overlap = 80
self.rag_batch_size = 16

# MORE...
# MUCH MORE...
# see settings_manager.py for all settings
```

Reasoning vs Working indicator (GPT‑5)
- gpt5_display_full_reasoning = true: Streams full reasoning summaries (gray) for GPT‑5 models while the model thinks; no generic “Working” indicator.
- gpt5_display_full_reasoning = false: Shows a generic “✦ Working” status after a short delay and keeps it until the first visible output; no reasoning text is shown.
- The “Working” indicator is automatically suppressed on turns where web search or deep search runs to avoid noisy output.

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