#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Markdown Translator
------------------
A tool for translating Markdown files to Korean, handling both single files and directories,
with special handling for large files by splitting them into manageable chunks.
"""

import os
import argparse
import glob
import re
from typing import List, Optional, Tuple
import anthropic
import time
from tqdm import tqdm
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("markdown-translator")

# Constants
MAX_CHUNK_SIZE = 8000  # Characters
OVERLAP_SIZE = 200     # Characters for context overlap


class MarkdownTranslator:
    """Class to handle translating markdown files to Korean."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the translator with API key."""
        # Load from .env file first
        load_dotenv()
        
        # Use provided key, then environment variable
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY not found in environment or .env file and not provided")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info("Translator initialized")
    
    def split_markdown(self, content: str) -> List[str]:
        """Split markdown content into manageable chunks while preserving structure."""
        if len(content) <= MAX_CHUNK_SIZE:
            return [content]
        
        chunks = []
        current_pos = 0
        
        # Try to split at markdown section boundaries where possible
        section_patterns = [
            r'\n## ', r'\n### ', r'\n#### ', r'\n##### ', r'\n###### ',
            r'\n\n', r'\n---\n', r'\n\*\*\*\n'
        ]
        
        while current_pos < len(content):
            # Determine end position for current chunk
            end_pos = min(current_pos + MAX_CHUNK_SIZE, len(content))
            
            # If we're not at the end of the content, try to find a good split point
            if end_pos < len(content):
                # Look for good split points
                best_split = end_pos
                for pattern in section_patterns:
                    # Find the last occurrence of the pattern before the max chunk size
                    content_slice = content[current_pos:end_pos]
                    matches = list(re.finditer(pattern, content_slice))
                    if matches:
                        last_match = matches[-1]
                        potential_split = current_pos + last_match.start() + 1
                        if potential_split > current_pos:  # Ensure we're making progress
                            best_split = potential_split
                            break
                
                end_pos = best_split
            
            # Extract the chunk
            chunk = content[current_pos:end_pos]
            chunks.append(chunk)
            
            # Move position for next chunk, with some overlap for context
            current_pos = max(current_pos, end_pos - OVERLAP_SIZE)
        
        logger.info(f"Split content into {len(chunks)} chunks")
        return chunks
    
    def translate_text(self, text: str) -> str:
        """Translate text to Korean using Claude."""
        retries = 3
        backoff = 2  # seconds
        
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Translate the following markdown text to Korean. 
Preserve all markdown formatting and structure exactly (including headings, lists, code blocks, links, etc.).
Only translate the actual content text, NOT code in code blocks, variable names, function names, 
placeholder text in brackets like {{variable}}, or technical terms that should remain in English.

Here is the markdown text to translate:

{text}"""
                        }
                    ]
                )
                
                translated_text = response.content[0].text
                return translated_text
            
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Translation failed: {e}. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                else:
                    logger.error(f"Translation failed after {retries} attempts: {e}")
                    raise
    
    def translate_markdown_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Translate a single markdown file to Korean."""
        # Determine output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_ko{ext}"
        
        logger.info(f"Translating {input_path} to {output_path}")
        
        # Read the markdown file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into manageable chunks if needed
        chunks = self.split_markdown(content)
        
        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(tqdm(chunks, desc="Translating chunks")):
            logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            translated_chunk = self.translate_text(chunk)
            translated_chunks.append(translated_chunk)
            # Small delay to avoid rate limiting
            if i < len(chunks) - 1:
                time.sleep(1)
        
        # Combine translated chunks
        if len(chunks) > 1:
            # If we split the file, we need to carefully rejoin to avoid duplication
            # in the overlapping regions. This is a simple approach and might need
            # refinement based on actual results.
            translated_content = translated_chunks[0]
            for i in range(1, len(translated_chunks)):
                translated_content += translated_chunks[i]
        else:
            translated_content = translated_chunks[0]
        
        # Write translated content to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        
        logger.info(f"Translation completed: {output_path}")
        return output_path
    
    def translate_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Translate all markdown files in a directory."""
        logger.info(f"Processing directory: {directory_path}")
        
        # Find all markdown files in the directory
        if recursive:
            pattern = os.path.join(directory_path, '**', '*.md')
            md_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(directory_path, '*.md')
            md_files = glob.glob(pattern)
        
        logger.info(f"Found {len(md_files)} markdown files")
        
        # Translate each file
        translated_files = []
        for md_file in tqdm(md_files, desc="Translating files"):
            try:
                # Skip files that already have "_ko" suffix
                if "_ko.md" in md_file:
                    logger.info(f"Skipping already translated file: {md_file}")
                    continue
                
                output_file = self.translate_markdown_file(md_file)
                translated_files.append(output_file)
            except Exception as e:
                logger.error(f"Error translating {md_file}: {e}")
        
        return translated_files


def main():
    """Main entry point for the translator script."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Translate Markdown files to Korean")
    
    # File or directory input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", help="Input markdown file to translate")
    input_group.add_argument("-d", "--directory", help="Directory containing markdown files to translate")
    
    # Optional arguments
    parser.add_argument("-o", "--output", help="Output file path (for single file translation)")
    parser.add_argument("-r", "--recursive", action="store_true", 
                        help="Process directories recursively (default: True)")
    parser.add_argument("-k", "--api-key", help="Anthropic API key (optional, defaults to environment variable)")
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = MarkdownTranslator(api_key=args.api_key)
    
    # Process the input
    if args.file:
        logger.info(f"Starting translation of file: {args.file}")
        translator.translate_markdown_file(args.file, args.output)
    else:
        logger.info(f"Starting translation of directory: {args.directory}")
        translator.translate_directory(args.directory, args.recursive)
    
    logger.info("Translation tasks completed")


if __name__ == "__main__":
    main()
