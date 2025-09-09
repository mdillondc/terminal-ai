# Timeline: Deterministic, Time‑Aware Recall for Long Conversations

## TODO: Refactor to RAG-Log Feature

**Timeline should be renamed and refactored into `--rag-log` command:**
- Current timeline feature is poorly suited for actual timeline-related queries (dates, chronological sequences)
- However, it excels as a RAG system using conversation log as the data source
- Should be integrated with existing RAG command structure (`--rag`, `--rag-rebuild`, etc.)
- Rename from `--timeline` to `--rag-log` to better reflect its purpose as conversation log retrieval
- Maintain existing functionality but position it correctly as a RAG feature, not a timeline feature
- For true timeline functionality, implement separate `--search-log` feature (see search-log.md)

---

TL;DR
Timeline is a provider‑agnostic memory scaffold that makes long conversations usable by injecting small, time‑aware recall from your own logs—so models don’t have to “remember” everything. It improves continuity (earliest/latest mentions, date lists, time spans) while keeping prompts lean. In 2025, it meaningfully reduces recency bias and “lost‑in‑the‑middle” effects on local models, but it’s not a guarantee of correctness and remains limited compared to frontier cloud models.

---

## Why Timeline Exists

Long, messy conversations regularly exceed what models can reliably handle, especially when running locally. Even with large context windows, models often:
- Overweight the last few turns (recency bias)
- Miss mid‑conversation facts (lost‑in‑the‑middle)
- Hallucinate sequences and dates
- Struggle with event‑anchored arithmetic (e.g., “days since X”)

These issues persist on consumer GPUs (e.g. RTX 4090) because:
- Quantized local models (7B–30B, even gpt-oss:120b) have weaker long‑context fidelity than frontier cloud models
- Long prompts degrade instruction following and arithmetic
- Tool injections (search, recall, etc.) can misroute the model if not very targeted

Timeline addresses this by:
- Using per‑message UTC epoch timestamps to reconstruct temporal order deterministically
- Injecting only the few earlier snippets that matter for the current turn
- Labeling recall with local time and adding a compact, derived summary when appropriate
- Including anti‑parroting instructions that nudge the model to synthesize, not echo

You retain privacy and control: your logs stay authoritative and local; Timeline adds context at call time without modifying history.

---

## What Timeline Does

- Deterministic “earliest” and “latest” queries
  - Earliest/first mention of a topic (topic‑aware)
  - Latest/most recent mention of a topic (topic‑aware)
- Deterministic “first thing I asked”
  - Earliest global user message in the conversation
- General recall (semantic)
  - Top‑K earlier snippets under a token budget, excluding the most recent turns to avoid redundancy
- Derived timeline summaries
  - Derived: start=YYYY‑MM‑DD; current=YYYY‑MM‑DD; days_elapsed=N; days_inclusive=M (computed from message timestamps)
- Deterministic date listing
  - “Tell me each date we have spoken” → all distinct dates (user+assistant), sorted
- Transparency line
  - A one‑liner showing which turns/snippets were injected (when enabled)
- Anti‑parroting directive
  - A final instruction in the recall block telling the model to answer concisely without copying the recall verbatim

---

## How It Works (High‑Level)

1) Per‑message timestamps
- Every message (system/user/assistant) is logged with a `timestamp` (UTC epoch seconds).
- Content stays unmodified; timestamps live in metadata.
- Labels shown to the model use your local time; calculations are done via epochs.

2) Small, provider‑agnostic injections
- On each turn, Timeline decides whether to inject recall based on the user’s prompt and (optionally) a lightweight intent classification.
- For temporal prompts (earliest/latest/when/dates), it selects minimal, relevant snippets and sorts them chronologically.
- For general recall prompts, it uses semantic search under a token budget and excludes the most recent turns to reduce redundancy.

3) Anti‑parroting safeguards
- The injected block ends with a short, explicit instruction to synthesize a concise answer and avoid copying the recall verbatim.

4) Provider‑specific embedding profile
- The embedding provider/model follows your active chat provider (cloud vs local) for privacy consistency. See settings for defaults and overrides.

---

## When To Use Timeline

Best for:
- Long, ongoing threads where local models normally run out of context but continuity matters
- Time‑anchored questions:
  - “What’s the earliest mention of <topic>?”
  - “What did we say most recently about <topic>?”
  - “Summarize the current timeline/situation”
  - “List all dates we spoke”
- Derived spans and date math that should be based on actual conversation timestamps, not a model’s memory

Not ideal for:
- Pure general‑knowledge questions (e.g., “Are watermelons sugar‑infused in country X?”)
  - Timeline recall should be bypassed; the model should just answer the question
- Short, single‑topic sessions where continuity is trivial

---

## Usage

- Command (toggle): `--timeline`
  - Enables/disables Timeline for the current session
- You can keep Timeline on during long sessions; intent routing (if enabled) should avoid injecting recall for unrelated, general‑knowledge prompts
- For time‑anchored prompts, ask explicitly (earliest, latest, dates, timeline) to trigger the right behavior
- For general‑knowledge prompts, be concise and specific; Timeline should step aside

Examples:
- “What’s the first thing I asked in this conversation?”
- “What did we discuss most recently about ‘project delta’?”
- “Summarize the current timeline for the database migration”
- “List each date we talked, in order”

---

## Commands and Settings

Command:
- `--timeline`
  - Toggle Timeline mode on/off for the current session (provider‑agnostic)

Key settings (configurable via `~/.config/terminal-ai/config`; defaults live in `settings_manager.py`):
- `timeline` (bool, default: False)
  - Toggle state for Timeline mode
- `timeline_top_k` (int, default: 6)
  - Maximum semantic recall snippets for general recall
- `timeline_exclude_recent` (int, default: 6)
  - Exclude the N most recent turns from general recall to reduce redundancy
- `timeline_max_tokens` (int, default: 2400)
  - Token budget for the injected recall block
- `timeline_quote_mode` (str, default: "verbatim")
  - Packing mode for recall (currently verbatim only)
- `timeline_transparency` (bool, default: True)
  - Print a one‑liner showing which turns/snippets were injected

Temporal intent classification:
- `timeline_intent_use_llm` (bool, default: True)
  - Use the active chat provider to classify temporal intent vs general knowledge
- `timeline_intent_user_window` (int, default: 3)
  - Number of recent user turns included for intent classification
- `timeline_intent_assistant_window` (int, default: 3)
  - Number of recent assistant turns included for intent classification
- `timeline_intent_temperature` (float, default: 0.1)
  - Temperature for intent classification (small budget)
- `timeline_intent_max_tokens` (int, default: 256)
  - Token budget for the structured classification output
- `timeline_temporal_pool_size` (int, default: 24)
  - Initial semantic pool size for earliest/latest selection before deterministic min/max by timestamp

Deterministic date listing:
- No extra settings. Trigger with prompts like:
  - “tell me each/all dates we have spoken”
  - “list all dates we spoke”

Notes:
- Per‑message timestamps (epoch UTC) are always logged, independent of provider or Timeline state
- Date labels shown to the model use local time by default

---

## Known Limitations and 2025 Expectations

What Timeline improves (but cannot guarantee):
- Better temporal grounding: it reduces—but does not eliminate—wrong “first/last mention” answers and hallucinated timelines
- More predictable recall: it injects small, relevant snippets instead of huge histories
- Less reliance on raw long‑context memory: it encodes time explicitly

What can still go wrong in 2025:
- Accuracy is not guaranteed
  - The model can misinterpret injected context, ignore instructions, or perform fuzzy arithmetic under long prompts
- Intent misrouting
  - Recall may be injected when the user’s question is unrelated (e.g., general knowledge)
- Anchor ambiguity
  - The “Derived” line uses conversation timestamps; if the question needs a specific event anchor (e.g., “days since the last X”), the model may still misapply the anchor
- Long‑prompt degradation
  - Very long prompts can push models into summary mode and degrade instruction following
- Model variability and quantization
- Token and latency overhead
  - Recall adds tokens and can slow down responses
  
Bottom line for 2025:
- Timeline provides meaningful gains in long‑conversation continuity and temporal correctness on local models
- It does not match the reliability of frontier cloud models (OpenAI/Google) on very long, complex threads
- It should be presented as a helpful scaffold rather than a silver bullet

---

## Best Practices

- Be explicit when you want temporal behavior
  - Use “earliest”, “latest”, “dates”, or “timeline” words in your prompt
- Ask for specific arithmetic when needed
  - Example: “Compute the number of days between the first mention of <topic> and today (based on our logs). Show the math.”
- Verify via transparency
  - When `timeline_transparency` is on, scan the one‑liner to see what was injected before trusting the answer
- Prefer concise questions for general knowledge
  - Short, direct questions reduce the chance of recall injection; Timeline should step aside
- Keep an eye on token budgets
  - Adjust `timeline_max_tokens`, `timeline_top_k`, and `timeline_exclude_recent` to balance completeness and latency
- Switch off Timeline when inappropriate
  - For pure general‑knowledge sessions, you can disable `--timeline` or rely on intent routing (if enabled) to avoid recall

---

## Privacy and Provider‑Agnostic Behavior

- Logs remain the authoritative source of truth (stored locally)
- Provider‑agnostic recall
  - Timeline works with local or cloud models; the embedded recall uses the active provider’s embedding stack
- Privacy alignment
  - If you use a local model (e.g. in this app it would be through Ollama), Timeline uses a local embedding provider by default for recall
  - If you use a cloud model, Timeline will use cloud embeddings unless you override settings
- To keep strict privacy, use a local provider and local embeddings

---

## FAQ

- Does Timeline change my logs?
  - No. It only reads them to compute recall; it injects transient context at call time.

- Does Timeline leak more data to the cloud?
  - If you’re using a cloud chat model, your prompt already goes to the cloud. Timeline adds small recall snippets from your local logs to that prompt. For strict privacy, use a local provider (Ollama), and nothing is sent to cloud.

- Why don’t you always inject the full history?
  - Full histories are slow, expensive (tokens), and unreliable; models do worse on long, noisy inputs. Timeline prioritizes small, relevant snippets with timestamps.
  
---

## Roadmap

- Stronger intent routing
  - More robust detection to avoid injecting recall for unrelated questions
- Event‑anchored derived spans
  - Better detection of event anchors (e.g., “last purchase,” “first occurrence”) for accurate day‑count math
- Smarter redundancy control
  - Improved deduplication and semantic grouping under the token budget
- Expanded formatting control
  - Tighter control over output style when users ask for lists, bullets, or explicit calculations

---

## Conclusion

Timeline is worth using today because it brings deterministic, time‑aware recall to long conversations—especially valuable for local models that struggle with long‑context fidelity. It improves continuity, reduces recency bias, and makes temporal queries (earliest/latest/dates/timelines) more reliable without flooding prompts.

At the same time, it’s important to set expectations: in 2025, local models on consumer hardware still lag frontier cloud providers in long‑conversation accuracy, event‑anchored arithmetic, and strict instruction following. Timeline narrows that gap but does not eliminate it.

Work will continue (when I have time) to harden intent routing, anchoring, and formatting control. The long‑term goal is clear: preserve privacy with local models while approaching cloud‑level coherence on complex, multi‑day threads. Timeline is a practical step in that direction, and it’s useful now—just not magic.