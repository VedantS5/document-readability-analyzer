import os
import re
import pandas as pd
import textstat
from bs4 import BeautifulSoup
import argparse
from tqdm import tqdm

def extract_paragraphs_from_markdown(file_path):
    """Extract plain text paragraphs from a markdown file with HTML spans."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Use BeautifulSoup to extract text from HTML spans
    soup = BeautifulSoup(content, 'html.parser')
    
    # Get all spans
    spans = [span.get_text().strip() for span in soup.find_all('span') if span.get_text().strip()]
    
    # Process spans to find paragraph boundaries
    paragraphs = []
    current_paragraph = ""
    
    for i, span in enumerate(spans):
        # Check if this span starts a new paragraph
        if re.match(r'^\d+\.', span) or span.startswith("Note:") or span.startswith("Appendix") or "Report of" in span or "The following" in span:
            # Save the previous paragraph if it exists
            if current_paragraph and len(current_paragraph.split()) > 5:
                paragraphs.append(current_paragraph)
            # Start a new paragraph
            current_paragraph = span
        else:
            # Add to the current paragraph
            current_paragraph += " " + span
    
    # Add the last paragraph
    if current_paragraph and len(current_paragraph.split()) > 5:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def calculate_readability_metrics(text):
    """Calculate various readability metrics for a text."""
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
            'text_standard': textstat.text_standard(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'avg_syllables_per_word': textstat.avg_syllables_per_word(text),
            'word_count': textstat.lexicon_count(text),
            'sentence_count': textstat.sentence_count(text),
            'difficult_words_count': textstat.difficult_words(text),
            'difficult_words_percentage': (textstat.difficult_words(text) / textstat.lexicon_count(text) * 100) if textstat.lexicon_count(text) > 0 else 0
        }
    except Exception as e:
        print(f"Error calculating metrics for text: {text[:50]}... - {str(e)}")
        return {metric: 0 for metric in [
            'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 
            'smog_index', 'automated_readability_index', 'coleman_liau_index',
            'dale_chall_readability_score', 'text_standard', 'avg_sentence_length',
            'avg_syllables_per_word', 'word_count', 'sentence_count',
            'difficult_words_count', 'difficult_words_percentage'
        ]}

def process_markdown_directory(directory_path, output_file='readability_report.csv'):
    """Process all markdown files in a directory and save readability metrics to CSV."""
    results = []
    
    # Get list of markdown files
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]
    
    print(f"Found {len(md_files)} markdown files to process")
    
    # Process each file
    for filename in tqdm(md_files, desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        paragraphs = extract_paragraphs_from_markdown(file_path)
        
        print(f"Extracted {len(paragraphs)} paragraphs from {filename}")
        
        # Process each paragraph
        for i, paragraph in enumerate(paragraphs):
            metrics = calculate_readability_metrics(paragraph)
            metrics['filename'] = filename
            metrics['paragraph_number'] = i + 1
            metrics['paragraph_text'] = paragraph
            results.append(metrics)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Readability report saved to {output_file}")
        
        # Generate summary statistics
        summary = df.drop(['filename', 'paragraph_number', 'paragraph_text', 'text_standard'], axis=1).describe()
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary.to_csv(summary_file)
        print(f"Summary statistics saved to {summary_file}")
    else:
        print("No paragraphs were extracted from the files.")

def main():
    parser = argparse.ArgumentParser(description='Calculate readability metrics for markdown files')
    parser.add_argument('input_dir', help='Directory containing markdown files')
    parser.add_argument('--output', '-o', default='readability_report.csv',
                      help='Output CSV file name (default: readability_report.csv)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
    
    process_markdown_directory(args.input_dir, args.output)

if __name__ == "__main__":
    main()

