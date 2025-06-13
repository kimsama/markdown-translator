# Issues Found in translator.py

## Critical Issue: Chunk Overlap Logic Bug

**Location:** Lines 165-172 in `translate_markdown_file()`

**Problem:**
```python
if len(chunks) > 1:
    translated_content = translated_chunks[0]
    for i in range(1, len(translated_chunks)):
        translated_content += translated_chunks[i]  # This concatenates ALL chunks!
```

**Issue:** The code simply concatenates all translated chunks without properly handling the `OVERLAP_SIZE` (200 characters). This means:
- Overlapping content gets duplicated in the output
- Content could grow exponentially with each chunk
- Could cause memory issues and extremely long processing times

**Impact:** For large files, this can cause the translation to hang indefinitely as the content keeps growing.

## Other Potential Hanging Issues

### 1. File Reading Without Encoding Fallback
**Location:** Line 148
```python
with open(input_path, 'r', encoding='utf-8') as f:
```

**Issue:** No fallback encoding handling. If the file has encoding issues, this could hang or fail silently.

### 2. No Timeout on API Calls
**Location:** Lines 111-124 in `translate_text()`

**Issue:** OpenAI API calls have no timeout parameter. Network issues could cause indefinite hanging.

### 3. Infinite Loop Risk in Chunking
**Location:** Lines 73-99 in `split_markdown()`

**Issue:** The line `current_pos = max(current_pos, end_pos - OVERLAP_SIZE)` could potentially create scenarios where `current_pos` doesn't advance properly, causing an infinite loop.

## Recommended Solutions

### Immediate Fix
Use the optimized version instead:
```bash
python translator_optimized.py -f "your_file.md" -v
```

The optimized version (`translator_optimized.py`) addresses these issues with:
- Proper chunk overlap handling
- Multiple encoding fallbacks
- Better error handling and timeouts
- Memory-efficient streaming processing
- Progress tracking to identify hanging issues

### Debug Current Issue
Check the log file to see where it's stuck:
```bash
tail -f translator.log
```

### Long-term Fix
The regular `translator.py` should be updated to fix the chunk concatenation logic or deprecated in favor of the optimized version.