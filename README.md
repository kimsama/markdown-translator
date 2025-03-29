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

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Anthropic API key in the `.env` file:
   ```
   # Edit the .env file in the project directory
   ANTHROPIC_API_KEY=your_api_key_here
   ```
   
   Alternatively, you can set it as an environment variable:
   ```
   # Windows
   set ANTHROPIC_API_KEY=your_api_key_here
   
   # Linux/Mac
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Translate a single file:

```bash
python translator.py -f path/to/file.md
```

The translated file will be saved as `path/to/file_ko.md`

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

### Provide API key via command line:

```bash
python translator.py -f path/to/file.md -k your_api_key_here
```

## Advanced Configuration

You can modify the following constants in the script to adjust the chunking behavior:

- `MAX_CHUNK_SIZE`: Maximum size of each chunk in characters (default: 8000)
- `OVERLAP_SIZE`: Size of the overlap between chunks for context (default: 200)

## Notes

- The translator skips files that already have the "_ko" suffix to avoid translating already translated files.
- Code blocks, variable names, function names, and technical terms are preserved in English.
- The translation uses Claude 3 Opus model for optimal quality and accuracy.

## License

MIT
