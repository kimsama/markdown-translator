#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Markdown Translator
------------------
A tool for translating Markdown files to Korean, handling both single files and directories,
with special handling for large files by splitting them into manageable chunks.
Memory-optimized version.
"""

import os
import argparse
import glob
import re
import gc
import time
import datetime
import math
import sys
from typing import List, Optional, Generator, Tuple
import openai
from tqdm import tqdm
import logging
import sys
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


class ProgressTracker:
    """Class to track and display progress information for translations."""
    
    def __init__(self, total_chunks: int, total_chars: int = 0, file_path: str = None):
        self.total_chunks = total_chunks
        self.processed_chunks = 0
        self.total_chars = total_chars
        self.processed_chars = 0
        self.start_time = datetime.datetime.now()
        self.last_update_time = self.start_time
        self.translation_times = []
        self.file_path = file_path
        self.file_name = os.path.basename(file_path) if file_path else "Unknown file"
        
        # Initialize progress bar
        self.pbar = tqdm(
            total=total_chunks,
            desc=f"Translating {self.file_name}",
            unit="chunk",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} chunks | {percentage:3.0f}% | ETA: {remaining}",
            file=sys.stdout,
            dynamic_ncols=True
        )
    
    def update(self, chunk_size: int = 1, chars_processed: int = 0):
        """Update progress after processing a chunk."""
        now = datetime.datetime.now()
        elapsed = (now - self.last_update_time).total_seconds()
        self.translation_times.append(elapsed)
        
        self.processed_chunks += chunk_size
        self.processed_chars += chars_processed
        
        # Keep only last 5 translation times for estimating ETA
        if len(self.translation_times) > 5:
            self.translation_times = self.translation_times[-5:]
        
        # Update progress bar
        self.pbar.update(chunk_size)
        
        # Display detailed progress info
        self._display_progress_info()
        
        self.last_update_time = now
    
    def _display_progress_info(self):
        """Display detailed progress information."""
        now = datetime.datetime.now()
        elapsed_total = (now - self.start_time).total_seconds()
        elapsed_min = int(elapsed_total // 60)
        elapsed_sec = int(elapsed_total % 60)
        
        # Calculate remaining time
        if self.processed_chunks > 0 and len(self.translation_times) > 0:
            avg_time_per_chunk = sum(self.translation_times) / len(self.translation_times)
            chunks_remaining = self.total_chunks - self.processed_chunks
            est_remaining_sec = chunks_remaining * avg_time_per_chunk
            est_remaining_min = int(est_remaining_sec // 60)
            est_remaining_sec = int(est_remaining_sec % 60)
        else:
            est_remaining_min, est_remaining_sec = 0, 0
        
        # Calculate percentage
        percentage = (self.processed_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0
        
        # Calculate processing speed
        chars_per_sec = self.processed_chars / elapsed_total if elapsed_total > 0 else 0
        
        # Log progress info
        progress_message = (
            f"Progress: {self.processed_chunks}/{self.total_chunks} chunks ({percentage:.1f}%) | "
            f"Elapsed: {elapsed_min}m {elapsed_sec}s | "
            f"ETA: {est_remaining_min}m {est_remaining_sec}s | "
            f"Speed: {chars_per_sec:.1f} chars/sec"
        )
        
        logger.info(progress_message)
        
        # Also print to console for better visibility
        print(f"\r{progress_message}", end="\r")
        sys.stdout.flush()
    
    def close(self):
        """Close the progress tracker and display final stats."""
        total_elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        total_min = int(total_elapsed // 60)
        total_sec = int(total_elapsed % 60)
        
        self.pbar.close()
        
        logger.info(f"\nTranslation completed in {total_min}m {total_sec}s")
        if self.total_chars > 0:
            chars_per_sec = self.processed_chars / total_elapsed if total_elapsed > 0 else 0
            logger.info(f"Average speed: {chars_per_sec:.1f} characters/second")
            logger.info(f"Total characters processed: {self.processed_chars}")
        
        # Also print to stdout for better visibility
        print(f"\nTranslation of {self.file_name} completed in {total_min}m {total_sec}s")


class MarkdownTranslator:
    """Class to handle translating markdown files to Korean."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the translator with API key."""
        # Load from .env file first
        load_dotenv()
        
        # Use provided key, then environment variable
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file and not provided")
        
        # Set the model to use
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Translator initialized with model: {self.model}")
    
    def read_file_in_chunks(self, file_path: str, chunk_size: int = 1024*1024) -> Generator[str, None, None]:
        """Read a file in chunks to avoid loading the entire file into memory."""
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
    
    def get_file_size(self, file_path: str) -> int:
        """Get the size of a file in bytes."""
        return os.path.getsize(file_path)
    
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
        """Translate text to Korean using OpenAI's GPT model."""
        retries = 3
        backoff = 2  # seconds
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator specializing in technical documentation. Translate markdown text to Korean while preserving all markdown formatting and structure exactly. Do not translate code blocks, variable names, function names, placeholder text in brackets, or technical terms."
                        },
                        {
                            "role": "user",
                            "content": f"Translate this markdown text to Korean:\n\n{text}"
                        }
                    ]
                )
                
                translated_text = response.choices[0].message.content
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
        """Translate a single markdown file to Korean with memory optimizations."""
        # Determine output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_ko{ext}"
        
        file_size = self.get_file_size(input_path)
        logger.info(f"Translating {input_path} to {output_path} (File size: {file_size/1024:.2f} KB)")
        
        # For large files, process in chunks to avoid memory issues
        if file_size > 1024 * 1024:  # If file is larger than 1MB
            logger.info("Large file detected, using streaming approach")
            with open(output_path, 'w', encoding='utf-8') as out_file:
                # Estimate total chunks for progress bar
                estimated_chunks = max(1, int(file_size / (MAX_CHUNK_SIZE * 0.8)))
                
                # Initialize progress tracker
                progress = ProgressTracker(estimated_chunks, file_size, input_path)
                
                # Read and process file in large blocks
                block_size = 1024 * 1024  # 1MB blocks for reading
                current_block = ""
                processed_chunks = 0
                
                for block in self.read_file_in_chunks(input_path, block_size):
                    current_block += block
                    
                    # Process as many complete chunks as possible from current block
                    while len(current_block) >= MAX_CHUNK_SIZE * 1.5:  # Ensure we have enough text to form a good chunk
                        chunks = self.split_markdown(current_block[:MAX_CHUNK_SIZE * 2])  # Process the beginning
                        chunk = chunks[0]  # Take just the first chunk
                        
                        logger.info(f"Translating chunk {processed_chunks+1} ({len(chunk)} chars)")
                        print(f"\nTranslating chunk {processed_chunks+1}/{estimated_chunks} ({len(chunk)} chars)")
                        translated_chunk = self.translate_text(chunk)
                        out_file.write(translated_chunk)
                        out_file.flush()  # Ensure content is written to disk
                        
                        # Remove the processed chunk, keeping some overlap
                        if len(chunk) > OVERLAP_SIZE:
                            current_block = current_block[len(chunk)-OVERLAP_SIZE:]
                        else:
                            current_block = current_block[len(chunk):]
                        
                        processed_chunks += 1
                        
                        # Update progress
                        progress.update(1, len(chunk))
                        print_progress_bar(processed_chunks, estimated_chunks, 
                                         prefix=f'Progress', 
                                         suffix=f'Chunk {processed_chunks}/{estimated_chunks}', 
                                         length=40)
                        
                        # Small delay to avoid rate limiting
                        time.sleep(1)
                        
                        # Force garbage collection
                        del translated_chunk
                        gc.collect()
                
                # Process any remaining text
                if current_block:
                    chunks = self.split_markdown(current_block)
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Translating final chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                        translated_chunk = self.translate_text(chunk)
                        out_file.write(translated_chunk)
                        
                        # Update progress
                        progress.update(1, len(chunk))
                        
                        # Small delay to avoid rate limiting
                        if i < len(chunks) - 1:
                            time.sleep(1)
                        
                        del translated_chunk
                        gc.collect()
                
                # Close progress tracker and display final stats
                progress.close()
        else:
            # For smaller files, use the original approach but with better memory management
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into manageable chunks if needed
            chunks = self.split_markdown(content)
            total_chars = len(content)
            del content  # Free memory
            gc.collect()
            
            # Initialize progress tracker
            progress = ProgressTracker(len(chunks), total_chars, input_path)
            
            # Open the output file for immediate writing
            with open(output_path, 'w', encoding='utf-8') as out_file:
                # Translate each chunk
                for i, chunk in enumerate(chunks):
                    logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                    print(f"\nTranslating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                    translated_chunk = self.translate_text(chunk)
                    
                    # Write each chunk as it's translated
                    out_file.write(translated_chunk)
                    out_file.flush()
                    
                    # Update progress
                    progress.update(1, len(chunk))
                    print_progress_bar(i+1, len(chunks),
                                     prefix=f'Progress',
                                     suffix=f'Chunk {i+1}/{len(chunks)}',
                                     length=40)
                    
                    # Small delay to avoid rate limiting
                    if i < len(chunks) - 1:
                        time.sleep(1)
                    
                    # Clean up memory
                    del chunk
                    del translated_chunk
                    gc.collect()
            
            # Close progress tracker
            progress.close()
            
            # Clean up
            del chunks
            gc.collect()
        
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
        
        # Filter out already translated files
        md_files = [f for f in md_files if "_ko.md" not in f]
        
        logger.info(f"Found {len(md_files)} markdown files to translate")
        print(f"\nFound {len(md_files)} markdown files to translate")
        
        # Create directory progress tracker
        dir_progress = tqdm(
            total=len(md_files),
            desc="Overall progress",
            unit="file", 
            file=sys.stdout,
            dynamic_ncols=True
        )
        
        # Translate each file
        translated_files = []
        for i, md_file in enumerate(md_files):
            try:
                print(f"\n[{i+1}/{len(md_files)}] Translating: {os.path.basename(md_file)}")
                output_file = self.translate_markdown_file(md_file)
                translated_files.append(output_file)
                dir_progress.update(1)
                
                # Force garbage collection after each file
                gc.collect()
            except Exception as e:
                logger.error(f"Error translating {md_file}: {e}")
                print(f"Error translating {md_file}: {e}")
        
        dir_progress.close()
        print(f"\nCompleted translating {len(translated_files)} out of {len(md_files)} files")
        logger.info(f"Completed translating {len(translated_files)} out of {len(md_files)} files")
        
        return translated_files


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """Create a text-based progress bar for console display."""
    if total == 0:
        return
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()


def main():
    """Main entry point for the translator script."""
    # Start time for overall execution
    start_time = datetime.datetime.now()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Declare global variables
    global MAX_CHUNK_SIZE
    
    parser = argparse.ArgumentParser(description="Translate Markdown files to Korean")
    
    # File or directory input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", help="Input markdown file to translate")
    input_group.add_argument("-d", "--directory", help="Directory containing markdown files to translate")
    
    # Optional arguments
    parser.add_argument("-o", "--output", help="Output file path (for single file translation)")
    parser.add_argument("-r", "--recursive", action="store_true", 
                        help="Process directories recursively (default: True)")
    parser.add_argument("-k", "--api-key", help="OpenAI API key (optional, defaults to environment variable)")
    parser.add_argument("-m", "--model", default="gpt-3.5-turbo",
                       help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--chunk-size", type=int, default=8000,
                       help=f"Maximum chunk size in characters (default: {MAX_CHUNK_SIZE})")
    
    args = parser.parse_args()
    
    # Update constants if provided
    if args.chunk_size:
        MAX_CHUNK_SIZE = args.chunk_size
    
    # Display startup information
    print("\n" + "=" * 60)
    print(f"Markdown Korean Translator (Memory-Optimized)")
    print("=" * 60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using model: {args.model}")
    print(f"Chunk size: {MAX_CHUNK_SIZE} characters")
    if args.file:
        file_size = os.path.getsize(args.file) / 1024  # KB
        print(f"Processing file: {args.file} ({file_size:.2f} KB)")
    elif args.directory:
        print(f"Processing directory: {args.directory} (Recursive: {args.recursive})")
    print("=" * 60 + "\n")
    
    # Initialize translator
    translator = MarkdownTranslator(api_key=args.api_key, model=args.model)
    
    # Process the input
    if args.file:
        logger.info(f"Starting translation of file: {args.file}")
        translator.translate_markdown_file(args.file, args.output)
    else:
        logger.info(f"Starting translation of directory: {args.directory}")
        translator.translate_directory(args.directory, args.recursive)
    
    # End time and total duration
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 60)
    print(f"Translation tasks completed")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 60)
    
    logger.info(f"Translation tasks completed. Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()
