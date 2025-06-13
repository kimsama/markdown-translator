# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python application for translating Markdown files to Korean while preserving formatting. It handles large files through intelligent chunking and offers both regular and memory-optimized versions.

## Development Commands

### Setup
```bash
# Setup virtual environment
bash setup-venv.sh         # Linux/Mac
setup-venv.bat             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Regular version
python translator.py -f file.md
python translator.py -d directory/ -r

# Memory-optimized version (recommended for large files)
python translator_optimized.py -f file.md -v
python translator_optimized.py -d directory/ -r --chunk-size 4000

# Using different API providers
python translator_optimized.py -f file.md --provider openai
python translator_optimized.py -f file.md --provider anthropic

# Convenience scripts
bash translate.sh -f file.md           # Linux/Mac
translate-optimized.bat -f file.md     # Windows
```

### Testing and Debugging
```bash
# Debug mode (no API calls, test file handling)
python translator_optimized.py -f file.md --debug

# Verbose logging
python translator_optimized.py -f file.md -v
```

## Architecture

### Core Components
- **MarkdownTranslator**: Main class handling translation workflow
- **ProgressTracker**: (Optimized version) Advanced progress reporting with ETA and memory tracking
- **Chunking System**: Intelligent content splitting that respects markdown structure

### Key Configuration
- `MAX_CHUNK_SIZE`: 8000 characters (adjustable via --chunk-size)
- `OVERLAP_SIZE`: 200 characters for context preservation
- `MAX_BLOCK_SIZE`: 20KB for file reading (optimized version)

### Translation Pipeline
1. **File Detection**: Multi-encoding support (UTF-8, UTF-8-SIG, Latin-1, CP1252)
2. **Content Chunking**: Section-aware splitting preferring headers (## , ### ) over hard breaks
3. **Translation Loop**: Sequential processing with retry logic and rate limiting
4. **Output Assembly**: Streaming write to avoid memory accumulation

### Version Differences
- **translator.py**: Basic version, loads entire file into memory
- **translator_optimized.py**: Production version with streaming I/O, sophisticated progress tracking, and constant memory usage

## Environment Configuration

### API Provider Selection
The application now supports both OpenAI and Anthropic APIs. Configure in `.env` file:

```bash
# Choose API provider
API_PROVIDER=openai  # or "anthropic"

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307

# General model override (optional)
MODEL=gpt-3.5-turbo
```

### Command Line Options
- API key: `-k your_key_here` (optional, defaults to environment variable)
- Provider: `--provider openai` or `--provider anthropic`
- Model: `-m model_name` (overrides environment defaults)

## Important Development Notes

- The optimized version is preferred for production use and handles large files efficiently
- Chunking logic respects markdown structure to maintain translation quality
- All file operations include comprehensive encoding fallback chains
- Progress tracking includes verbose/quiet modes for different use cases
- Error handling includes exponential backoff for API calls and multiple fallback strategies
- Both OpenAI and Anthropic APIs are supported with automatic provider detection
- Default models: gpt-3.5-turbo (OpenAI), claude-3-haiku-20240307 (Anthropic)