# program to add <|endoftext|> separator at sentence boundary in national corpus data as it has
# - been randomized at sentence level so this will let Qwen dynamically adjust context weighting 
# - during pretraining.

import stanza
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time

SEP = "<|endoftext|>"

# Global variable to store the nlp pipeline to avoid re-initialization
nlp = None


# stanza has an Irish tokenizer so can tokenize at sentence level instead of crude methods.
def get_nlp_pipeline():
    global nlp
    if nlp is None:
        try:
            nlp = stanza.Pipeline('ga', verbose=False)
        except:
            # If the model isn't downloaded, download it first
            stanza.download('ga')
            nlp = stanza.Pipeline('ga', verbose=False)
    return nlp

# Add the separator at the end of the sentence
def add_separator(text, separator):
    if not text or not text.strip():
        return separator
    
    # Get the nlp pipeline
    nlp_pipeline = get_nlp_pipeline()
    
    # Process the input text
    doc = nlp_pipeline(text)
    sentences = [sentence.text.strip() for sentence in doc.sentences if sentence.text.strip()]    # Add separator at the end of each sentence
    if sentences:
        sentences_with_separator = [sentence + separator for sentence in sentences]
        new_text = "".join(sentences_with_separator)
    else:
        new_text = separator
    
    return new_text

# previous crude approach
def add_separator_simple(text, separator):
    if not text or not text.strip():
        return separator
    
    # Replace newlines with the separator
    result = text.replace('\n', separator)
    
    # Ensure it ends with the separator if it doesn't already
    if not result.endswith(separator):
        result += separator
    
    return result

import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# batch processing for memory constraints
def process_text_batch(text_batch, separator, batch_id):
    print(f"Processing batch {batch_id}...")
    start_time = time.time()
    
    result = add_separator_simple(text_batch, separator)
    
    end_time = time.time()
    print(f"Batch {batch_id} completed in {end_time - start_time:.2f} seconds")
    
    return result

def split_text_into_batches(text, batch_size=100000):
    if len(text) <= batch_size:
        return [text]
    
    batches = []
    start = 0
    
    while start < len(text):
        end = start + batch_size
        
        # If we're not at the end, try to break at a newline to avoid splitting sentences
        if end < len(text):
            # Look for the last newline within the batch
            last_newline = text.rfind('\n', start, end)
            if last_newline > start:
                end = last_newline + 1
        
        batches.append(text[start:end])
        start = end
    
    return batches

# Write to output, add SEP to source file name
def write_processed_file(input_file_path, output_file_path, separator, use_threading=True, max_workers=4, batch_size=100000):
    print(f"Reading file: {input_file_path}")
    
    # Read the input file
    with open(input_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"File size: {len(raw_text):,} characters")
    
    if use_threading and len(raw_text) > batch_size:
        print(f"Using threading with {max_workers} workers and batch size {batch_size:,}")
        
        # Split text into batches
        batches = split_text_into_batches(raw_text, batch_size)
        print(f"Split into {len(batches)} batches")
        
        # Process batches with threading
        processed_batches = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing
            futures = []
            for i, batch in enumerate(batches):
                future = executor.submit(process_text_batch, batch, separator, i+1)
                futures.append(future)
            
            # Collect results in order
            for future in futures:
                processed_batches.append(future.result())
        
        # Combine all processed batches
        processed_text = "".join(processed_batches)
        
    else:
        print("Processing without threading")
        processed_text = add_separator_simple(raw_text, separator)
    
    # Write the processed text to output file
    print(f"Writing processed file: {output_file_path}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(processed_text)
    
    print(f"Processing complete!")
    print(f"Original size: {len(raw_text):,} characters")
    print(f"Processed size: {len(processed_text):,} characters")
    print(f"Output file: {output_file_path}")

# run
if __name__ == "__main__":
    input_file = "./data/DCU.txt"
    
    # Create output filename by adding _SEP before the extension
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_SEP{input_path.suffix}"
    
    print(f"Processing {input_file} -> {output_file}")
    
    # Process the file with threading enabled
    write_processed_file(
        input_file_path=input_file,
        output_file_path=str(output_file),
        separator=SEP,
        use_threading=True,
        max_workers=4,
        batch_size=50000  # 50K characters per batch
    )
    

