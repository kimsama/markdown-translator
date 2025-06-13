# Markdown Korean Translator

A Python application for translating Markdown files to Korean using OpenAI or Anthropic APIs. This tool preserves markdown formatting while translating content, and handles large files by splitting them into manageable chunks.

## Features

- Translate individual markdown files to Korean
- Support for both OpenAI and Anthropic APIs
- Process an entire directory of markdown files at once
- Preserve all markdown formatting and structure
- Handle large files by automatically splitting them into chunks
- Output files with '_ko' suffix

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/ai-translater.git
   cd ai-translater
   ```

2. Set up a virtual environment (recommended):
   ```
   # On Windows
   setup-venv.bat
   
   # On Linux/Mac
   bash setup-venv.sh
   ```
   
   Or manually:
   ```
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure your API provider and keys in the `.env` file:
   ```
   # Copy .env.example to .env and edit:
   cp .env.example .env
   
   # Choose your preferred API provider
   API_PROVIDER=openai  # or "anthropic"
   
   # For OpenAI (get key from https://platform.openai.com/)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   
   # For Anthropic (get key from https://console.anthropic.com/)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ANTHROPIC_MODEL=claude-3-haiku-20240307
   ```
   
   Alternatively, you can set environment variables:
   ```
   # Windows
   set API_PROVIDER=openai
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export API_PROVIDER=anthropic
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Translate a single file:

```bash
# Using default provider from .env
python translator_optimized.py -f path/to/file.md

# Using specific API provider
python translator_optimized.py -f path/to/file.md --provider openai
python translator_optimized.py -f path/to/file.md --provider anthropic
```

The translated file will be saved as `path/to/file_ko.md`

The translator handles large files efficiently by:
- Processing files in smaller chunks
- Reducing memory usage
- Adding robust error handling
- Writing translated content to disk immediately

### Translate with a specific output path:

```bash
python translator_optimized.py -f path/to/file.md -o path/to/output.md
```

### Translate all markdown files in a directory:

```bash
python translator_optimized.py -d path/to/directory
```

This will translate all `.md` files in the directory and save them with the '_ko' suffix.

### Translate recursively (including subdirectories):

```bash
python translator_optimized.py -d path/to/directory -r
```

### Run in debug mode without performing actual translations:

```bash
python translator_optimized.py -f path/to/file.md --debug
```

Debug mode processes the file through all steps but doesn't make any API calls for translation. This is useful for testing file handling, especially with large files, without incurring API costs. The output file will contain placeholder text instead of actual translations.

### Control logging verbosity:

```bash
python translator_optimized.py -f path/to/file.md -v
```

By default, the translator shows only a clean progress bar with minimal distraction. This "quiet mode" hides processing details, warnings about chunk splitting, and other technical logs.

Use the `-v` or `--verbose` flag to see detailed logs about:
- File reading and encoding detection
- Markdown chunk splitting and processing
- Warnings about potential issues
- Detailed progress information

The verbose mode is useful for debugging or monitoring the translation process in detail.

### Provide API key via command line:

```bash
# OpenAI
python translator_optimized.py -f path/to/file.md --provider openai -k your_openai_key

# Anthropic
python translator_optimized.py -f path/to/file.md --provider anthropic -k your_anthropic_key
```

### Specify a different model:

```bash
# OpenAI models
python translator_optimized.py -f path/to/file.md --provider openai -m gpt-4
python translator_optimized.py -f path/to/file.md --provider openai -m gpt-4o

# Anthropic models
python translator_optimized.py -f path/to/file.md --provider anthropic -m claude-3-5-sonnet-20241022
python translator_optimized.py -f path/to/file.md --provider anthropic -m claude-3-5-haiku-20241022
```

## Advanced Configuration

You can modify the following constants in the script to adjust the chunking behavior:

- `MAX_CHUNK_SIZE`: Maximum size of each chunk in characters (default: 8000)
- `OVERLAP_SIZE`: Size of the overlap between chunks for context (default: 200)

### Additional Options

The translator offers additional customization via command line:

```bash
# Change the chunk size (in characters)
python translator_optimized.py -f path/to/file.md --chunk-size 4000

# Run in debug mode (test without making API calls)
python translator_optimized.py -f path/to/file.md --debug

# Specify API provider and model
python translator_optimized.py -f path/to/file.md --provider anthropic -m claude-3-5-sonnet-20241022
```

The optimized version also includes:

- `MAX_BLOCK_SIZE`: Default block size for reading files (20KB)
- `MIN_BLOCK_SIZE`: Minimum block size for very large files (4KB)

These values are automatically adjusted based on file size but can be modified in the script if needed.

## API Providers

### OpenAI
- **Default model**: gpt-3.5-turbo
- **Other models**: gpt-4, gpt-4o, gpt-4-turbo-preview
- **Get API key**: https://platform.openai.com/

### Anthropic
- **Default model**: claude-3-haiku-20240307
- **Other models**: claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022
- **Get API key**: https://console.anthropic.com/
- **Note**: Anthropic models appear to be better at preserving code blocks untranslated

## Notes

- The translator skips files that already have the "_ko" suffix to avoid translating already translated files.
- Code blocks, variable names, function names, and technical terms are preserved in English.
- Both OpenAI and Anthropic APIs are supported with automatic provider detection.
- Default models are automatically selected based on the chosen provider.

## License

MIT
