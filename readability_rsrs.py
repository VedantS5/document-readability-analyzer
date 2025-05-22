import os
import re
import pandas as pd
import textstat
from bs4 import BeautifulSoup
import argparse
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

# Add this line to download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class RSRSCalculator:
    def __init__(self, model_name="roberta-base"):
        """Initialize the RSRS calculator with a language model."""
        print(f"Loading language model {model_name} for RSRS calculation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        print("Language model loaded.")
        
        
    def calculate_rsrs(self, text):
        """Calculate Ranked Sentence Readability Score for text."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
            
        sentence_scores = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Calculate word negative log-likelihoods (WNLL)
            wnlls = self._get_word_nlls(sentence)
            
            # Sort words by difficulty (measured by WNLL)
            sorted_wnlls = sorted(wnlls)
            
            # Calculate RSRS with rank weighting
            if sorted_wnlls:
                rsrs = sum(np.sqrt(i+1) * score for i, score in enumerate(sorted_wnlls)) / len(sorted_wnlls)
                sentence_scores.append(rsrs)
        
        # Return document RSRS as average of sentence scores
        return sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0
    
    def _get_word_nlls(self, sentence):
        """Get negative log-likelihood for each word in a sentence."""
        tokens = self.tokenizer(sentence, return_tensors="pt", return_special_tokens_mask=True)
        input_ids = tokens["input_ids"]
        special_tokens_mask = tokens["special_tokens_mask"]
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            
        # Get log probabilities
        logits = outputs.logits[0, :-1]  # all except last token
        target_ids = input_ids[0, 1:]    # all except first token
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get negative log-likelihood for each token
        token_nlls = []
        for i, target_id in enumerate(target_ids):
            if special_tokens_mask[0, i+1] == 0:  # Skip special tokens
                token_nlls.append(-log_probs[i, target_id].item())
        
        return token_nlls

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

def calculate_readability_metrics(text, rsrs_calculator=None):
    """Calculate various readability metrics for a text."""
    try:
        metrics = {
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
        
        # Add RSRS if calculator is provided
        if rsrs_calculator is not None:
            try:
                metrics['rsrs_score'] = rsrs_calculator.calculate_rsrs(text)
            except Exception as e:
                print(f"Error calculating RSRS for text: {text[:50]}... - {str(e)}")
                metrics['rsrs_score'] = 0
                
        return metrics
    except Exception as e:
        print(f"Error calculating metrics for text: {text[:50]}... - {str(e)}")
        metrics = {metric: 0 for metric in [
            'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 
            'smog_index', 'automated_readability_index', 'coleman_liau_index',
            'dale_chall_readability_score', 'text_standard', 'avg_sentence_length',
            'avg_syllables_per_word', 'word_count', 'sentence_count',
            'difficult_words_count', 'difficult_words_percentage'
        ]}
        if rsrs_calculator is not None:
            metrics['rsrs_score'] = 0
        return metrics

def process_markdown_directory(directory_path, output_file='readability_report.csv', include_rsrs=False, model_name="roberta-base"):
    """Process all markdown files in a directory and save readability metrics to CSV."""
    results = []
    
    # Initialize RSRS calculator if needed
    rsrs_calculator = None
    if include_rsrs:
        rsrs_calculator = RSRSCalculator(model_name)
    
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
            metrics = calculate_readability_metrics(paragraph, rsrs_calculator)
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
        summary_cols = [col for col in df.columns if col not in ['filename', 'paragraph_number', 'paragraph_text', 'text_standard']]
        summary = df[summary_cols].describe()
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
    parser.add_argument('--rsrs', action='store_true',
                      help='Include RSRS (Ranked Sentence Readability Score) calculation')
    parser.add_argument('--model', default='roberta-base',
                      help='Language model to use for RSRS calculation (default: roberta-base)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
    
    process_markdown_directory(args.input_dir, args.output, args.rsrs, args.model)

if __name__ == "__main__":
    main()

