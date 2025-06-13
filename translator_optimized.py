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
import anthropic
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

# Set httpx and other third-party loggers to a higher level to silence them by default
for module in ['httpx', 'urllib3', 'openai', 'anthropic']:
    logging.getLogger(module).setLevel(logging.WARNING)

# Constants
MAX_CHUNK_SIZE = 8000  # Characters
OVERLAP_SIZE = 200     # Characters for context overlap
MAX_BLOCK_SIZE = 20 * 1024  # Default block size for reading (20KB)
MIN_BLOCK_SIZE = 4 * 1024   # Minimum block size for very large files (4KB)

# Global variables
VERBOSE_MODE = False  # Controls logging output verbosity

# Remove existing handlers to reconfigure
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a file handler that logs everything with detailed format
file_handler = logging.FileHandler("translator.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Create a custom formatter that only shows the message content without any prefixes
class SimpleFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

# Add a console handler with simplified formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings and errors by default
console_handler.setFormatter(SimpleFormatter())
logger.addHandler(console_handler)

# Create a custom filter to control warning messages
class WarningFilter(logging.Filter):
    def filter(self, record):
        # In non-verbose mode, filter out all warning messages related to processing
        if not VERBOSE_MODE and record.levelno == logging.WARNING:
            # Filter out ANY warning containing these strings
            warning_patterns = [
                "Current block too large",
                "Detected potential infinite loop",
                "File reading stuck at position"
            ]
            for pattern in warning_patterns:
                if pattern in record.msg:
                    return False
        return True

# Apply the filter to the console handler
console_handler.addFilter(WarningFilter())

def set_verbose_mode(verbose: bool):
    """Set the verbose mode for logging output."""
    global VERBOSE_MODE
    VERBOSE_MODE = verbose
    
    # Adjust console logging level based on verbose mode
    if verbose:
        console_handler.setLevel(logging.INFO)  # Show INFO level logs in console
        # In verbose mode, allow httpx and other libraries logs
        for module in ['httpx', 'urllib3', 'openai', 'anthropic']:
            logging.getLogger(module).setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)  # Only show warnings and errors
        # In non-verbose mode, silence httpx and other libraries logs
        for module in ['httpx', 'urllib3', 'openai', 'anthropic']:
            logging.getLogger(module).setLevel(logging.WARNING)
        
    logger.info(f"Verbose mode is {'enabled' if verbose else 'disabled'}")

def log_info(message: str):
    """Log info message, considering verbose mode settings."""
    # These specific messages should only be logged to the file in verbose mode
    translation_progress_logs = [
        "Read chunk", 
        "Processing block with", 
        "Current block size after appending", 
        "Splitting current block", 
        "Found good split point", 
        "Adding chunk of size", 
        "Moving to position", 
        "Split content into", 
        "Got first chunk with",
        "Processing chunk from position",
        "Splitting markdown content of",
        "Detected potential infinite loop",
        "Trying to read file with encoding",
        "Reading chunk",
        "Finished reading file",
        "Content fits in a single chunk",
        "Falling back to simple chunk splitting",
        "Translating chunk"
    ]
    
    # For certain technical logs, only log them if in verbose mode
    if not VERBOSE_MODE and any(substr in message for substr in translation_progress_logs):
        # Only log to file, skip logging completely for these technical messages
        return
    
    # For all other messages, log normally
    logger.info(message)  # Always log to file
    
    # Skip printing to console in normal mode, let tqdm handle the progress display
    # Console handler will handle this based on level

def log_warning(message: str):
    """Log warning message, respecting verbose mode for certain types of warnings."""
    # These specific warnings should only be logged to the file in non-verbose mode
    verbose_only_warnings = [
        "Detected potential infinite loop",
        "File reading stuck at position",
        "Current block too large"
    ]
    
    # If not in verbose mode and the message contains any of the patterns,
    # only log to file and skip console completely
    if not VERBOSE_MODE and any(pattern in message for pattern in verbose_only_warnings):
        # Skip ALL console output for these warnings in non-verbose mode
        # Only write to the log file
        silent_log(message, level="WARNING")
        return

    # For all other cases, use the normal logging
    logger.warning(message)

def silent_log(message: str, level: str = "INFO"):
    """Log directly to file without console output to avoid breaking progress display."""
    with open("translator.log", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        log_file.write(f"{timestamp} - markdown-translator - {level} - {message}\n")

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
        
        # Initialize progress bar with format based on verbose mode
        if VERBOSE_MODE:
            bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} chunks | {percentage:3.0f}% | ETA: {remaining}"
        else:
            # Simpler format for non-verbose mode
            bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
            
        self.pbar = tqdm(
            total=total_chunks,
            desc=f"Translating {self.file_name}",
            unit="chunk",
            bar_format=bar_format,
            file=sys.stdout,
            dynamic_ncols=False,  # Prevent resizing which causes redrawing
            position=0,           # Fix the position to prevent multiple progress bars
            leave=True,           # Keep the progress bar after completion
            miniters=1,           # Update display minimal number of times
            disable=None,         # Don't disable even if not a TTY
            ncols=100             # Fix width to prevent resizing issues
        )
        
        # Record initial display to ensure we maintain a single progress bar
        # This is important to "register" our progress bar
        self.pbar.refresh()
    
    def update_after_chunk(self, chunk_size: int, total_chunks_processed: int):
        """Update progress tracking after processing a chunk."""
        # Update progress display
        if VERBOSE_MODE:
            # In verbose mode, clear the line before updating progress
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()
        
        # Safety check - if we have more chunks than estimated, update the total
        if total_chunks_processed > self.total_chunks:
            # Cap the maximum increase to prevent extreme values
            max_increase = min(total_chunks_processed + 5, int(self.total_chunks * 1.5))
            self.total_chunks = max_increase
            self.pbar.total = self.total_chunks
            log_info(f"Increased estimated chunks to {self.total_chunks} (processing chunk {total_chunks_processed})")
        
        # Track progress percentage for logging
        old_percentage = (self.pbar.n / self.total_chunks * 100) if self.total_chunks > 0 else 0
        
        # Advanced progress display algorithm for smoother progress bar updates
        # Calculate the ideal display position based on current state
        if total_chunks_processed >= self.total_chunks:
            # We've processed all chunks, show 100%
            display_position = self.total_chunks
        elif self.processed_chunks == 0 and total_chunks_processed == 1:
            # First chunk - show initial progress
            display_position = 1
        else:
            # For intermediate chunks, calculate a smooth progression
            
            # Detect if this is likely the start of final chunks processing
            # (when more than 30% progress jumped in one step)
            if total_chunks_processed > self.processed_chunks + 1:
                # We're in the final chunks phase
                if VERBOSE_MODE:
                    log_info(f"Detected final processing phase, adapting progress display")
                
                # Calculate how many chunks we expect to process in total
                # (current + estimated remaining based on content size)
                percentage_complete = self.processed_chars / self.total_chars if self.total_chars > 0 else 0.5
                if percentage_complete > 0.8:
                    # We're near the end, distribute remaining progress more aggressively
                    display_position = int(self.total_chunks * 0.85) + (
                        (total_chunks_processed - self.processed_chunks) / 2
                    )
                else:
                    # More gradual progression in the middle
                    current_percentage = percentage_complete * 100
                    target_percentage = min(current_percentage + 10, 90)  # Increment by up to 10%
                    display_position = int(self.total_chunks * (target_percentage / 100))
            elif total_chunks_processed > 0.7 * self.total_chunks:
                # We're approaching the end of regular processing
                # Slow down progress to leave room for final chunks
                remaining_chunks = self.total_chunks - total_chunks_processed
                display_position = self.total_chunks - remaining_chunks * 1.2
            else:
                # Normal progression during regular processing
                display_position = total_chunks_processed
        
        # Ensure display position is within bounds
        display_position = max(0, min(display_position, self.total_chunks))
        display_position = int(display_position)  # Ensure it's an integer
        
        # Only update if there's actual progress to show
        current_display = self.pbar.n
        if display_position > current_display or total_chunks_processed >= self.total_chunks:
            self.pbar.n = display_position
            self.pbar.refresh()
            
            # Log significant progress changes
            new_percentage = (display_position / self.total_chunks * 100) if self.total_chunks > 0 else 0
            if int(new_percentage / 10) > int(old_percentage / 10):
                if VERBOSE_MODE:
                    log_info(f"Progress update: {new_percentage:.1f}% (chunk {total_chunks_processed}/{self.total_chunks})")
        
        # Also update detailed progress info for logging
        self.processed_chunks = total_chunks_processed
        self.processed_chars += chunk_size
        self._display_progress_info()
    
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
        
        # Build progress message - only log to file
        progress_message = (
            f"Progress: {self.processed_chunks}/{self.total_chunks} chunks ({percentage:.1f}%) | "
            f"Elapsed: {elapsed_min}m {elapsed_sec}s | "
            f"ETA: {est_remaining_min}m {est_remaining_sec}s | "
            f"Speed: {chars_per_sec:.1f} chars/sec"
        )
        
        # Only log to file, never to console - let tqdm handle the visual output completely
        if VERBOSE_MODE:
            logger.info(progress_message)
    
    def close(self):
        """Close the progress tracker and display final stats."""
        total_elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        total_min = int(total_elapsed // 60)
        total_sec = int(total_elapsed % 60)
        
        # Ensure progress is set to 100% before closing
        self.pbar.n = self.total_chunks
        self.pbar.refresh()
        self.pbar.close()
        
        # Clear the line to ensure clean output
        if not VERBOSE_MODE:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()
        
        # Log completion details
        #completion_message = f"\nTranslation completed in {total_min}m {total_sec}s"
        #logger.info(completion_message)
        
        if self.total_chars > 0:
            chars_per_sec = self.processed_chars / total_elapsed if total_elapsed > 0 else 0
            stats_message = f"Average speed: {chars_per_sec:.1f} characters/second"
            logger.info(stats_message)
            logger.info(f"Total characters processed: {self.processed_chars}")
        
        # Always show completion message, regardless of verbose mode
        print(f"\nTranslation of {self.file_name} completed in {total_min}m {total_sec}s")


class MarkdownTranslator:
    """Class to handle translating markdown files to Korean."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, debug_mode: bool = False, api_provider: Optional[str] = None):
        """Initialize the translator with API key and provider."""
        # Load from .env file first
        load_dotenv()
        
        # Determine API provider
        self.api_provider = api_provider or os.environ.get("API_PROVIDER", "openai").lower()
        
        # Set default models based on provider
        if self.api_provider == "anthropic":
            default_model = "claude-3-haiku-20240307"
            api_key_env = "ANTHROPIC_API_KEY"
        else:
            default_model = "gpt-3.5-turbo"
            api_key_env = "OPENAI_API_KEY"
        
        # Use provided key, then environment variable
        if api_key is None:
            api_key = os.environ.get(api_key_env)
            if api_key is None:
                raise ValueError(f"{api_key_env} not found in environment or .env file and not provided")
        
        # Set the model to use - check provider-specific env vars first
        if model:
            self.model = model
        elif self.api_provider == "anthropic":
            self.model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("MODEL", default_model)
        else:
            self.model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL", default_model)
        
        # Debug mode for testing
        self.debug_mode = debug_mode
        if debug_mode:
            logger.info("Running in DEBUG mode - no actual translation will be performed")
        
        # Initialize the appropriate client
        if self.api_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = openai.OpenAI(api_key=api_key)
        
        logger.info(f"Translator initialized with {self.api_provider} provider and model: {self.model}")
    
    def _safe_open_output_file(self, file_path: str, mode: str = 'w'):
        """Open an output file with safe encoding handling."""
        try:
            return open(file_path, mode, encoding='utf-8')
        except Exception as e:
            logger.error(f"Error opening output file with utf-8 encoding: {e}")
            try:
                return open(file_path, mode, encoding='utf-8-sig')
            except Exception as e:
                logger.error(f"Error opening output file with utf-8-sig encoding: {e}")
                # Last resort, try with default system encoding
                return open(file_path, mode)
    
    def read_file_in_chunks(self, file_path: str, chunk_size: int = 1024*1024) -> Generator[str, None, None]:
        """Read a file in chunks to avoid loading the entire file into memory."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                log_info(f"Trying to read file with encoding: {encoding}")
                with open(file_path, 'r', encoding=encoding) as f:
                    chunk_count = 0
                    last_position = -1
                    while True:
                        log_info(f"Reading chunk {chunk_count+1} from file")
                        current_position = f.tell()
                        
                        # Safety check for infinite loop
                        if current_position == last_position:
                            # Use direct file logging in non-verbose mode
                            if not VERBOSE_MODE:
                                silent_log(f"File reading stuck at position {current_position}, breaking", level="WARNING")
                            else:
                                log_warning(f"File reading stuck at position {current_position}, breaking")
                            break
                        
                        last_position = current_position
                        data = f.read(chunk_size)
                        if not data:
                            log_info(f"Finished reading file, read {chunk_count} chunks with encoding {encoding}")
                            break
                        chunk_count += 1
                        log_info(f"Read chunk {chunk_count} with {len(data)} characters ({encoding})")
                        yield data
                # If we get here, we successfully read the file
                return
            except UnicodeDecodeError as e:
                logger.error(f"Error reading file {file_path} with encoding {encoding}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error reading file {file_path} with encoding {encoding}: {e}")
                logger.exception("Exception details:")
                # Try the next encoding
                continue
        
        # If we get here, we failed with all encodings
        error_msg = f"Could not read file {file_path} with any of the attempted encodings: {encodings_to_try}"
        logger.error(error_msg)
        raise IOError(error_msg)
    
    def get_file_size(self, file_path: str) -> int:
        """Get the size of a file in bytes."""
        return os.path.getsize(file_path)
    
    def split_markdown(self, content: str) -> List[str]:
        """Split markdown content into manageable chunks while preserving structure."""
        try:
            log_info(f"Splitting markdown content of {len(content)} characters")
            if len(content) <= MAX_CHUNK_SIZE:
                log_info("Content fits in a single chunk, returning as is")
                return [content]
            
            chunks = []
            current_pos = 0
            last_end_pos = -1  # Track the last ending position to detect infinite loops
            
            # Try to split at markdown section boundaries where possible
            section_patterns = [
                r'\n## ', r'\n### ', r'\n#### ', r'\n##### ', r'\n###### ',
                r'\n\n', r'\n---\n', r'\n\*\*\*\n'
            ]
            
            while current_pos < len(content):
                # Determine end position for current chunk
                end_pos = min(current_pos + MAX_CHUNK_SIZE, len(content))
                log_info(f"Processing chunk from position {current_pos} to {end_pos}")
                
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
                                log_info(f"Found good split point at position {best_split} using pattern {pattern}")
                                break
                    
                    end_pos = best_split
                
                # Extract the chunk
                chunk = content[current_pos:end_pos]
                log_info(f"Adding chunk of size {len(chunk)} characters")
                chunks.append(chunk)
                
                # Check for infinite loop - if we're not making progress, force move forward
                if end_pos <= last_end_pos:
                    # Use silent logging instead of direct file access
                    if not VERBOSE_MODE:
                        silent_log(f"Detected potential infinite loop at position {current_pos}. Forcing move forward.", level="WARNING")
                    else:
                        log_warning(f"Detected potential infinite loop at position {current_pos}. Forcing move forward.")
                    # Force move beyond the current position
                    current_pos = last_end_pos + MAX_CHUNK_SIZE // 2
                else:
                    # Move position for next chunk, with some overlap for context
                    next_pos = max(current_pos, end_pos - OVERLAP_SIZE)
                    # Ensure we're making progress (at least 100 characters)
                    if next_pos < current_pos + 100:
                        next_pos = current_pos + 100
                    current_pos = next_pos
                
                last_end_pos = end_pos
                log_info(f"Moving to position {current_pos} for next chunk (with overlap)")
            
            log_info(f"Split content into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error in split_markdown: {e}")
            logger.exception("Exception details:")
            # Fallback: if something goes wrong, just split by size
            try:
                log_info("Falling back to simple chunk splitting")
                chunks = []
                current_pos = 0
                while current_pos < len(content):
                    end_pos = min(current_pos + MAX_CHUNK_SIZE, len(content))
                    chunks.append(content[current_pos:end_pos])
                    current_pos = end_pos  # Ensure we're making progress
                log_info(f"Split content into {len(chunks)} chunks using fallback method")
                return chunks
            except Exception as nested_e:
                logger.error(f"Error in fallback splitting: {nested_e}")
                # Ultimate fallback: return the whole content
                return [content]
    
    def translate_text(self, text: str) -> str:
        """Translate text to Korean using the configured API provider."""
        # In debug mode, just return the original text
        if self.debug_mode:
            logger.info("DEBUG MODE: Skipping actual translation")
            return f"[DEBUG MODE] This would be translated: {text[:100]}..."
            
        retries = 3
        backoff = 2  # seconds
        
        for attempt in range(retries):
            try:
                if self.api_provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=[
                            {
                                "role": "user",
                                "content": f"You are a professional translator specializing in technical documentation. Translate markdown text to Korean while preserving all markdown formatting and structure exactly. Do not translate code blocks, variable names, function names, placeholder text in brackets, or technical terms.\n\nTranslate this markdown text to Korean:\n\n{text}"
                            }
                        ]
                    )
                    translated_text = response.content[0].text
                else:
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
        try:
            # Determine output path if not provided
            if output_path is None:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}_ko{ext}"
            
            file_size = self.get_file_size(input_path)
            # This is an important status message, keep it visible in all modes
            #logger.info(f"Translating {input_path} to {output_path} (File size: {file_size/1024:.2f} KB)")
            #print(f"Translating {os.path.basename(input_path)} (File size: {file_size/1024:.2f} KB)")
            
            # For large files, process in chunks to avoid memory issues
            if file_size > 20 * 1024:  # If file is larger than 200KB
                log_info("Large file detected, using streaming approach")
                try:
                    # Estimate total chunks for progress bar - IMPORTANT: keep fixed throughout
                    # Use a conservative estimation approach for markdown files
                    # We multiply by a larger factor since markdown formatting takes up more space
                    # and we need to account for the chunking logic at section boundaries
                    estimated_chunks = max(10, math.ceil(file_size / (MAX_CHUNK_SIZE * 0.5)))
                    log_info(f"Estimated number of chunks: {estimated_chunks}")
                    
                    # Track actual chunks processed - both main chunks and final chunks
                    total_chunks_processed = 0
                    
                    # Initialize progress tracker
                    progress = ProgressTracker(estimated_chunks, file_size, input_path)
                    
                    # Adjust block size based on file size
                    # For very large files, use smaller blocks to avoid memory issues
                    if file_size > 1024 * 1024:  # > 1MB
                        block_size = MIN_BLOCK_SIZE  # 4KB for very large files
                        overlap_size = 100  # Smaller overlap for large files
                    else:
                        block_size = MAX_BLOCK_SIZE  # 20KB for medium files
                        overlap_size = OVERLAP_SIZE  # Normal overlap
                    
                    if VERBOSE_MODE:
                        log_info(f"Using block size of {block_size} bytes and overlap of {overlap_size} chars for reading")
                    current_block = ""
                    processed_chunks = 0
                    
                    # Open output file for writing
                    with self._safe_open_output_file(output_path, 'w') as out_file:
                        #log_info("Starting to process file in blocks")
                        try:
                            for block in self.read_file_in_chunks(input_path, block_size):
                                log_info(f"Processing block with {len(block)} characters")
                                current_block += block
                                log_info(f"Current block size after appending: {len(current_block)} characters")
                                
                                # Process as many complete chunks as possible from current block
                                while len(current_block) >= MAX_CHUNK_SIZE:  # Process when we have at least one chunk worth of content
                                    log_info(f"Splitting current block (size: {len(current_block)}) into chunks")
                                    chunks = self.split_markdown(current_block[:int(MAX_CHUNK_SIZE * 1.5)])  # Process the beginning
                                    chunk = chunks[0]  # Take just the first chunk
                                    log_info(f"Got first chunk with {len(chunk)} characters")
                                    
                                    log_info(f"Translating chunk {processed_chunks+1} ({len(chunk)} chars)")
                                    if VERBOSE_MODE:
                                        # In verbose mode, print on the same line without disturbing the progress bar
                                        sys.stdout.write(f"\rTranslating chunk {processed_chunks+1}/{estimated_chunks} ({len(chunk)} chars)")
                                        sys.stdout.flush()
                                    translated_chunk = self.translate_text(chunk)
                                    # Add a newline if the translated chunk doesn't end with one
                                    if translated_chunk and not translated_chunk.endswith('\n'):
                                        translated_chunk += '\n'
                                    out_file.write(translated_chunk)
                                    out_file.flush()  # Ensure content is written to disk
                                    
                                    # Remove the processed chunk, keeping some overlap
                                    if len(chunk) > overlap_size:
                                        current_block = current_block[len(chunk)-overlap_size:]
                                    else:
                                        current_block = current_block[len(chunk):]
                                    
                                    # Force garbage collection after each chunk
                                    processed_chunks += 1
                                    gc.collect()
                                    
                                    # Update our total count of processed chunks
                                    total_chunks_processed += 1
                                    
                                    # Use the improved update_after_chunk method to handle progress
                                    progress.update_after_chunk(len(chunk), total_chunks_processed)
                                    
                                    # Small delay to avoid rate limiting
                                    time.sleep(1)
                                    
                                    # Clean up memory
                                    del chunk
                                    del translated_chunk
                                    gc.collect()
                                    
                                    # Break out if block gets too big to avoid memory issues
                                    if len(current_block) > MAX_CHUNK_SIZE * 2:
                                        # Use silent logging in non-verbose mode
                                        if not VERBOSE_MODE:
                                            silent_log("Current block too large, breaking to process next block", level="WARNING")
                                        else:
                                            log_warning("Current block too large, breaking to process next block")
                                        break
                            
                            # Process any remaining text in the current block
                            if current_block:
                                # Use silent_log to avoid interrupting progress bar updates
                                silent_log(f"Processing remaining text block of size {len(current_block)}")
                                
                                chunks = self.split_markdown(current_block)
                                
                                # Silently log without console output to avoid breaking the progress bar
                                chunk_size_total = sum(len(c) for c in chunks)
                                silent_log(f"Processing final chunks: {len(chunks)} chunks ({chunk_size_total} chars) remaining")

                                # Calculate progress silently
                                chars_processed_so_far = progress.processed_chars
                                total_chars_estimate = chars_processed_so_far + chunk_size_total
                                
                                # Update progress with more accurate total char count - log only to file
                                if total_chars_estimate > progress.total_chars and progress.total_chars > 0:
                                    progress.total_chars = total_chars_estimate
                                    silent_log(f"Updated total character count to {progress.total_chars}")
                                
                                # Pre-announce the final processing stage - no console output
                                if len(chunks) > 0:
                                    # This triggers the detection of final phase in update_after_chunk
                                    progress.processed_chunks = total_chunks_processed - 2
                                
                                # Ensure we have enough "space" in the progress bar for the remaining chunks - silent logging
                                if total_chunks_processed + len(chunks) > progress.total_chunks:
                                    # Increase the total to accommodate all final chunks plus a small buffer
                                    adjusted_total = total_chunks_processed + len(chunks) + 2
                                    silent_log(f"Adjusted total chunks to {adjusted_total} for final processing")
                                
                                for i, chunk in enumerate(chunks):
                                    # Write to log file directly without console output to avoid breaking progress bar
                                    silent_log(f"Translating final chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                                    
                                    # Only show the "Translating final chunk" message in verbose mode
                                    if VERBOSE_MODE:
                                        sys.stdout.write(f"\rTranslating final chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                                        sys.stdout.flush()
                                    
                                    translated_chunk = self.translate_text(chunk)
                                    # Add a newline if the translated chunk doesn't end with one
                                    if translated_chunk and not translated_chunk.endswith('\n'):
                                        translated_chunk += '\n'
                                    out_file.write(translated_chunk)
                                    out_file.flush()  # Ensure content is written to disk
                                    
                                    # Update our total count of processed chunks
                                    total_chunks_processed += 1
                                    
                                    # Use the improved update_after_chunk method to handle progress
                                    progress.update_after_chunk(len(chunk), total_chunks_processed)
                                    
                                    # Small delay to avoid rate limiting - but not for the last chunk
                                    if i < len(chunks) - 1:
                                        time.sleep(1)
                                    
                                    # Clean up memory
                                    del chunk
                                    del translated_chunk
                                    gc.collect()
                        
                        except Exception as e:
                            logger.error(f"Error processing blocks: {e}")
                            logger.exception("Exception details:")
                            print(f"Error processing blocks: {e}")
                            raise
                    
                    # Close progress tracker and display final stats outside of the file handling
                    # Ensure progress shows 100% at the end
                    progress.pbar.n = progress.total_chunks
                    progress.pbar.refresh()
                    progress.close()
                
                except Exception as e:
                    logger.error(f"Error in translate_markdown_file: {e}")
                    logger.exception("Exception details:")
                    print(f"Error processing file: {e}")
                    raise
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
                
                # Track chunks processed
                total_chunks_processed = 0
                
                # Open the output file for immediate writing
                with self._safe_open_output_file(output_path, 'w') as out_file:
                    # Translate each chunk - with better progress tracking
                    for i, chunk in enumerate(chunks):
                        log_info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                        if VERBOSE_MODE:
                            sys.stdout.write(f"\rTranslating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                            sys.stdout.flush()
                        translated_chunk = self.translate_text(chunk)
                        
                        # Write each chunk as it's translated
                        # Add a newline if the translated chunk doesn't end with one
                        if translated_chunk and not translated_chunk.endswith('\n'):
                            translated_chunk += '\n'
                        out_file.write(translated_chunk)
                        out_file.flush()
                        
                        # Update our total count of processed chunks
                        total_chunks_processed += 1
                        
                        # Use the improved update_after_chunk method to handle progress
                        progress.update_after_chunk(len(chunk), total_chunks_processed)
                        
                        # Small delay to avoid rate limiting - but not for the last chunk
                        if i < len(chunks) - 1:
                            time.sleep(1)
                        
                        # Clean up memory
                        del chunk
                        del translated_chunk
                        gc.collect()
                
                # Close progress tracker
                # Ensure progress shows 100% at the end
                progress.pbar.n = progress.total_chunks
                progress.pbar.refresh()
                progress.close()
                
                # Clean up
                del chunks
                gc.collect()
            
            logger.info(f"Translation completed: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in translate_markdown_file: {e}")
            logger.exception("Exception details:")
            print(f"Error processing file: {e}")
            raise
    
    def translate_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Translate all markdown files in a directory."""
        # This is an important status message, keep it visible in all modes
        logger.info(f"Processing directory: {directory_path}")
        print(f"Processing directory: {directory_path}")
        
        # Find all markdown files in the directory
        if recursive:
            pattern = os.path.join(directory_path, '**', '*.md')
            md_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(directory_path, '*.md')
            md_files = glob.glob(pattern)
        
        # Filter out already translated files
        md_files = [f for f in md_files if "_ko.md" not in f]
        
        # This is important info, keep it visible
        logger.info(f"Found {len(md_files)} markdown files to translate")
        print(f"\nFound {len(md_files)} markdown files to translate")
        
        # Create directory progress tracker
        dir_progress = tqdm(
            total=len(md_files),
            desc="Overall progress",
            unit="file", 
            file=sys.stdout,
            dynamic_ncols=False,  # Prevent resizing which causes redrawing
            position=0,           # Fix the position to prevent multiple progress bars
            leave=True,           # Keep the progress bar after completion
            miniters=max(1, len(md_files) // 10),  # Update display less frequently 
            disable=None,         # Don't disable even if not a TTY
            ncols=100             # Fix width to prevent resizing issues
        )
        
        # Translate each file
        translated_files = []
        for i, md_file in enumerate(md_files):
            try:
                # Important status update for file progress
                if VERBOSE_MODE:
                    sys.stdout.write(f"\r[{i+1}/{len(md_files)}] Translating: {os.path.basename(md_file)}      ")
                    sys.stdout.flush()
                output_file = self.translate_markdown_file(md_file)
                translated_files.append(output_file)
                
                # Just update the counter without refreshing display
                dir_progress.n = i + 1
                
                # Only refresh at certain intervals or at the end
                if i + 1 == len(md_files) or (i + 1) % max(1, len(md_files) // 5) == 0:
                    dir_progress.refresh()
                
                # Force garbage collection after each file
                gc.collect()
            except Exception as e:
                logger.error(f"Error translating {md_file}: {e}")
                print(f"Error translating {md_file}: {e}")
        
        # Ensure progress is set to 100% before closing
        dir_progress.n = len(md_files)
        dir_progress.refresh()
        dir_progress.close()
        
        # Clear the line to ensure clean output
        if not VERBOSE_MODE:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()
        
        # Final summary - always visible
        print(f"\nCompleted translating {len(translated_files)} out of {len(md_files)} files")
        logger.info(f"Completed translating {len(translated_files)} out of {len(md_files)} files")
        
        return translated_files


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
    parser.add_argument("-k", "--api-key", help="API key (optional, defaults to environment variable)")
    parser.add_argument("-m", "--model", 
                       help="Model to use (defaults: gpt-3.5-turbo for OpenAI, claude-3-haiku-20240307 for Anthropic)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                       help="API provider to use (default: openai)")
    parser.add_argument("--chunk-size", type=int, default=8000,
                       help=f"Maximum chunk size in characters (default: {MAX_CHUNK_SIZE})")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode without actual translation (for testing)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging output")
    
    args = parser.parse_args()
    
    # Update constants if provided
    if args.chunk_size:
        MAX_CHUNK_SIZE = args.chunk_size
    
    # Set verbose mode
    set_verbose_mode(args.verbose)
    
    # Display startup information
    print("\n" + "=" * 60)
    print(f"Markdown Korean Translator (Memory-Optimized)")
    print("=" * 60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using provider: {args.provider}")
    print(f"Using model: {args.model or ('gpt-3.5-turbo' if args.provider == 'openai' else 'claude-3-haiku-20240307')}")
    print(f"Chunk size: {MAX_CHUNK_SIZE} characters")
    if args.file:
        file_size = os.path.getsize(args.file) / 1024  # KB
        print(f"Processing file: {args.file} ({file_size:.2f} KB)")
    elif args.directory:
        print(f"Processing directory: {args.directory} (Recursive: {args.recursive})")
    print("=" * 60 + "\n")
    
    # Initialize translator
    translator = MarkdownTranslator(api_key=args.api_key, model=args.model, debug_mode=args.debug, api_provider=args.provider)
    
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
    
    #logger.info(f"Translation tasks completed. Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()
