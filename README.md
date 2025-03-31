# Markdown Korean Translator

A Python application for translating Markdown files to Korean. This tool preserves markdown formatting while translating content, and handles large files by splitting them into manageable chunks.

## Features

- Translate individual markdown files to Korean
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

3. Set your OpenAI API key in the `.env` file:
   ```
   # Edit the .env file in the project directory
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo  # or another model of your choice
   ```
   
   Alternatively, you can set it as an environment variable:
   ```
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   set OPENAI_MODEL=gpt-3.5-turbo
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   export OPENAI_MODEL=gpt-3.5-turbo
   ```

## Usage

### Translate a single file:

```bash
python translator.py -f path/to/file.md
```

The translated file will be saved as `path/to/file_ko.md`

## Memory-Optimized Version

For large files or when processing many files, you can use the memory-optimized version:

```bash
python translator_optimized.py -f path/to/file.md
```

The optimized version handles large files more efficiently by:
- Processing files in smaller chunks
- Reducing memory usage
- Adding more robust error handling
- Writing translated content to disk immediately

All commands that work with `translator.py` also work with `translator_optimized.py`.

### Translate with a specific output path:

```bash
python translator.py -f path/to/file.md -o path/to/output.md
```

### Translate all markdown files in a directory:

```bash
python translator.py -d path/to/directory
```

This will translate all `.md` files in the directory and save them with the '_ko' suffix.

### Translate recursively (including subdirectories):

```bash
python translator.py -d path/to/directory -r
```

### Run in debug mode without performing actual translations:

```bash
python translator.py -f path/to/file.md --debug
```

Debug mode processes the file through all steps but doesn't make any API calls for translation. This is useful for testing file handling, especially with large files, without incurring OpenAI API costs. The output file will contain placeholder text instead of actual translations.

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
python translator.py -f path/to/file.md -k your_api_key_here
```

### Specify a different OpenAI model:

```bash
python translator.py -f path/to/file.md -m gpt-4
```

## Advanced Configuration

You can modify the following constants in the script to adjust the chunking behavior:

- `MAX_CHUNK_SIZE`: Maximum size of each chunk in characters (default: 8000)
- `OVERLAP_SIZE`: Size of the overlap between chunks for context (default: 200)

### Additional Options in the Optimized Version

The optimized version offers additional customization via command line:

```bash
# Change the chunk size (in characters)
python translator_optimized.py -f path/to/file.md --chunk-size 4000

# Run in debug mode (test without making API calls)
python translator_optimized.py -f path/to/file.md --debug
```

The optimized version also includes:

- `MAX_BLOCK_SIZE`: Default block size for reading files (20KB)
- `MIN_BLOCK_SIZE`: Minimum block size for very large files (4KB)

These values are automatically adjusted based on file size but can be modified in the script if needed.

## Notes

- The translator skips files that already have the "_ko" suffix to avoid translating already translated files.
- Code blocks, variable names, function names, and technical terms are preserved in English.
- The translation uses OpenAI's GPT-3.5-turbo model by default, but you can specify other models.

## License

MIT
