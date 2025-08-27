#!/bin/bash

# Terminal AI Helper Script Example
#
# This script demonstrates how to create powerful AI workflow shortcuts by sourcing
# helper functions in your .bashrc or .zshrc. It transforms multi-step terminal-ai
# commands into simple, memorable shortcuts.
#
# Usage:
#   X. IMPORTANT! Update TERMINAL_AI_PATH variable below to your actual path
#
#   1. Customize the model variables below for your preferred models
#   2. Source this script: source ~/path/to/shell.example.sh
#   3. Run 'ai' to see all available shortcuts
#   4. Use shortcuts like: s "hello", sw "latest AI news", cb "explain this", su "https://www.youtube.com/watch?v=6mp_CGzx6p4" to summarize video, etc
#
# Benefits:
#   - Turn long commands into shortcuts
#   - Consistent model switching without remembering model names
#   - Reusable workflows for common AI tasks
#   - Quick access to specialized AI modes (search, clipboard, url, youtube, etc)
#
# Zsh Tip: Add 'setopt nonomatch' to your ~/.zshrc to enable unquoted arguments:
#   s hello world        # Instead of s "hello world"
#   sw latest AI news    # Instead of sw "latest AI news"
#
# Clipboard URL Feature: The URL functions (u, ul, su, sul) can automatically
# fetch URLs from clipboard if no URL is provided. This requires clipboard utilities:
#   - Wayland: wl-paste (install: sudo pacman -S wl-clipboard / sudo apt install wl-clipboard)
#   - X11: xclip (install: sudo pacman -S xclip / sudo apt install xclip)
#   - macOS: pbpaste (built-in, no installation needed)

# Configure your preferred models here
export AI_MODEL=gemini-2.5-flash       # Fast, cost-effective model for general use
export AI_MODEL_LOCAL=hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K_XL  # Local model for privacy-sensitive tasks
export TERMINAL_AI_PATH=~/Sync/scripts/terminal-ai  # Update this to your actual terminal-ai path

# Display all available AI shortcuts with descriptions (auto-detected)
ai() {
  echo "Terminal AI Helper Functions:"
  echo "============================="
  # export AI_MODEL=gpt-5-mini
  export AI_MODEL=gemini-2.5-flash
  export AI_MODEL_LOCAL=hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q6_K_XL
  # export AI_MODEL_LOCAL=gpt-oss:20b

  # Help function that extracts usage from actual functions
  ai() {
    for func in s sl sw swl swd swdl cb cbl u ul su sul cli clil rewrite rewritel us24 us24l auditArch auditArchl auditAuth auditAuthl; do
      local help=$(declare -f "$func" | grep -o 'help="[^"]*"' | cut -d'"' -f2)
      printf "%-8s %s\n" "$func:" "$help"
    done
  }

  # Helper function to handle common AI operations
  _ai() {
    local model="$1"
    shift 1

    # Smart argument detection
    if [[ "$1" == --* ]]; then
      # Second argument is a flag/command, treat as extra_args
      local extra_args="$1"
      shift 1
      local query="$*"
    else
      # Second argument is part of query, no extra_args
      local extra_args=""
      local query="$*"
    fi

    conda activate terminal-ai
    cd ~/Sync/scripts/terminal-ai || {
      echo "Failed to cd into script directory"
      return 1
    }

    if [ -z "$query" ]; then
      python src/main.py --input "--model $model $extra_args"
    else
      python src/main.py --input "--model $model $extra_args" --input "$query"
    fi
  }

  # Get all function names and group cloud/local pairs together
  local all_funcs=($(typeset -f | grep -E '^[a-z][a-z0-9]*[[:space:]]*\(\)' | grep -v '^ai[[:space:]]*\(\)' | grep -v '^_' | sed 's/[[:space:]]*().*$//' | sort))

  # Group functions by base name (cloud first, then local)
  local funcs=()
  local processed=()

  for func in "${all_funcs[@]}"; do
    # Skip if already processed
    [[ " ${processed[*]} " =~ " $func " ]] && continue

    # Add cloud version
    funcs+=("$func")
    processed+=("$func")

    # Add local version if it exists
    local local_func="${func}l"
    if [[ " ${all_funcs[*]} " =~ " $local_func " ]]; then
      funcs+=("$local_func")
      processed+=("$local_func")
    fi
  done

  for func_name in "${funcs[@]}"; do
    # Extract help string using typeset -f which works in both shells
    local help=$(typeset -f "$func_name" 2>/dev/null | grep -o 'help="[^"]*"' | head -1 | cut -d'"' -f2)
    if [[ -n "$help" ]]; then
      printf "%-10s %s\n" "$func_name:" "$help"
    fi
  done
}

# Helper function for URL operations with automatic clipboard detection
_url() {
  local model="$1"

  # Safe argument handling - check if we have enough arguments before shifting
  if [ $# -lt 2 ]; then
    # No URL provided, everything after model is query
    local url=""
    shift 1
    local query="$*"
  else
    local url="$2"
    shift 2
    local query="$*"
  fi

  # Check if URL is valid (starts with http:// or https://)
  if [[ -z "$url" || ! "$url" =~ ^https?:// ]]; then
    # If we had invalid text as URL, treat it as part of the query
    if [ -n "$url" ]; then
      query="$url $query"
    fi
    # Try to get URL from clipboard (supports Wayland, X11, and macOS)
    url=$(wl-paste 2>/dev/null || xclip -selection clipboard -o 2>/dev/null || pbpaste 2>/dev/null)
    if [[ -z "$url" || ! "$url" =~ ^https?:// ]]; then
      echo "Error: No valid URL provided or found in clipboard"
      return 1
    fi
    echo "Using URL from clipboard: $url"
  fi

  conda activate terminal-ai
  cd "$TERMINAL_AI_PATH" || {
    echo "Failed to cd into script directory"
    return 1
  }

  if [ -n "$query" ]; then
    python src/main.py --input "--model $model --url $url" --input "$query"
  else
    python src/main.py --input "--model $model --url $url"
  fi
}

# Smart AI helper function that detects command flags automatically
_ai() {
  local model="$1"
  shift 1

  # Smart argument detection - if 2nd arg starts with --, it's a flag
  if [[ "$1" == --* ]]; then
    local extra_args="$1"
    shift 1
    local query="$*"
  else
    local extra_args=""
    local query="$*"
  fi

  # Navigate to terminal-ai directory and activate environment
  conda activate terminal-ai
  cd "$TERMINAL_AI_PATH" || {
    echo "Failed to cd into script directory"
    return 1
  }

  if [ -z "$query" ]; then
    python src/main.py --input "--model $model $extra_args"
  else
    python src/main.py --input "--model $model $extra_args" --input "$query"
  fi
}

s() {
  local help="Basic AI conversation with cloud model"
  _ai "$AI_MODEL" "$*"
}

sl() {
  local help="Local (Ollama): Basic AI conversation"
  _ai "$AI_MODEL_LOCAL" "$*"
}

s5() {
  local help="Basic AI conversation with cloud model"
  _ai "gpt-5" "$*"
}

sw() {
  local help="Web search + AI analysis - great for current events and research"
  _ai "$AI_MODEL" "--search" "$*"
}

swl() {
  local help="Local (Ollama): Web search + AI analysis"
  _ai "$AI_MODEL_LOCAL" "--search" "$*"
}

swd() {
  local help="Deep web search + comprehensive AI analysis - for complex research"
  _ai "$AI_MODEL" "--search-deep" "$*"
}

swdl() {
  local help="Local (Ollama): Deep web search + comprehensive AI analysis"
  _ai "$AI_MODEL_LOCAL" "--search-deep" "$*"
}

cb() {
  local help="Analyze clipboard content"
  _ai "$AI_MODEL" "--cb" "$*"
}

cbl() {
  local help="Local (Ollama): Analyze clipboard content"
  _ai "$AI_MODEL_LOCAL" "--cb" "$*"
}

u() {
  local help="Summarize URL content (paste URL or use clipboard)"
  _url "$AI_MODEL" "$@"
}

ul() {
  local help="Local (Ollama): Summarize URL content"
  _url "$AI_MODEL_LOCAL" "$@"
}

su() {
  local help="Quick URL summary (3 key bullet points)"
  _url "$AI_MODEL" "$*" "summarize the most important points in 1 to 3 bullet points."
}

sul() {
  local help="Local (Ollama): Quick URL summary"
  ul "$*" "summarize the most important points in 1 to 3 bullet points."
}

cli() {
  local help="CLI/system administration expert mode"
  _ai "$AI_MODEL" "--instructions cli.md" "$*"
}

clil() {
  local help="Local (Ollama): CLI/system administration expert mode"
  _ai "$AI_MODEL_LOCAL" "--instructions cli.md" "$*"
}

rewrite() {
  local help="Rewrite/improve text from clipboard"
  _ai "$AI_MODEL" "--instructions rewrite.md --cb" "Improve this text"
}

rewritel() {
  local help="Local (Ollama): Rewrite/improve text from clipboard"
  _ai "$AI_MODEL_LOCAL" "--instructions rewrite.md --cb" "Improve this text"
}

# Add this script to your shell profile:
# echo "source ~/path/to/shell.example.sh" >> ~/.bashrc   # for bash
# echo "source ~/path/to/shell.example.sh" >> ~/.zshrc    # for zsh

# Automatically source personal script if it exists (excluded from git)
# Personal script is excluded from git to keep private/custom functions separate
# You should can just remove this, it's just for my own (terminal-ai author) benefit
if [[ -n "$ZSH_VERSION" ]]; then
  # zsh compatibility
  [[ -f "${${(%):-%x}:A:h}/shell.personal.sh" ]] && source "${${(%):-%x}:A:h}/shell.personal.sh" 2>/dev/null || true
else
  # bash compatibility
  [[ -f "$(dirname "${BASH_SOURCE[0]}")/shell.personal.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/shell.personal.sh" 2>/dev/null || true
fi