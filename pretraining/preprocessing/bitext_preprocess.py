
def write_bitext():
    # read ./data2/ELRC_ga.txt and ELREC_en.txt files, they are aligned by line
    # concatenate with language tags <en> "line from ELREC_en.txt" <ga> "line from ELRC_ga.txt" <|endoftext|>
    # write to ./data ELRC_bitext
    with open ("./data2/ELRC_ga.txt", "r", encoding="utf-8") as ga_file, \
        open("./data2/ELRC_en.txt", "r", encoding="utf-8") as en_file, \
        open("./data/ELRC_bitext.txt", "w", encoding="utf-8") as output_file:

        for ga_line, en_line in tqdm(zip(ga_file, en_file)):
            # Strip whitespace and add language tags
            ga_line = ga_line.strip()
            en_line = en_line.strip()

            # remove HTML entities and spacing around punctuation
            # spacing was used in MT, but we want natural text for LM
            ga_line = clean_text(ga_line)
            en_line = clean_text(en_line)
            # Write the concatenated line to the output file
            bitext_line = f"<en> {en_line} <ga> {ga_line} <|endoftext|>"
            output_file.write(bitext_line)

import html
import re
from tqdm import tqdm

def clean_text(text):
    # Fix spaced HTML entities like &amp; quot ;
    text = re.sub(r'&\s*([a-z]+)\s*;', r'&\1;', text)
    # Decode HTML entities (even double-escaped)
    text = html.unescape(html.unescape(text))
    # Fix spaces around punctuation
    text = re.sub(r'\s+([.,;:!?()])', r'\1', text)

    # remove space after opening brackets
    text = re.sub(r'([([])\s+', r'\1', text)  

    # remove trailing comma
    text = text.rstrip(',')
    return text

if __name__ == "__main__":
    print("Processing ELRC bitext...")
    write_bitext()
    print("ELRC bitext written to ./data/ELRC_bitext.txt")
    


