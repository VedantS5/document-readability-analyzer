# Document Readability Analyzer 📚

A comprehensive toolkit for analyzing document readability using both traditional readability metrics and advanced neural language model-based techniques.

## Overview

This toolkit provides three main components:

1. **Basic Readability Analysis** (`readability_textstat.py`): Analyzes text documents using traditional readability metrics like Flesch-Kincaid, Gunning Fog, and more.

2. **Advanced RSRS Analysis** (`readability_rsrs.py`): Implements the Ranked Sentence Readability Score (RSRS) using transformer-based language models to evaluate text complexity based on word prediction difficulty.

3. **DOCX to Markdown Converter** (`docx_to_md.py`): Converts Microsoft Word documents to markdown format for easy readability analysis.

## Features

### Traditional Readability Metrics
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Automated Readability Index
- Coleman-Liau Index
- Dale-Chall Readability Score
- Sentence and word count statistics
- Difficult words percentage

### Advanced Neural Metrics
- Ranked Sentence Readability Score (RSRS)
- Uses transformer models to evaluate text complexity
- Word-level difficulty scoring based on predictability
- Rank-weighted scoring to emphasize difficult words

### Document Processing
- Extract paragraphs from markdown files with HTML spans
- Process entire directories of documents
- Generate comprehensive CSV reports with metrics
- Statistical summaries of document readability

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/readability.git
   cd readability
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Readability Analysis

```bash
python readability_textstat.py <input_directory> [--output readability_report.csv]
```

Arguments:
- `input_directory`: Directory containing markdown files to analyze
- `--output`: Optional output CSV file name (default: readability_report.csv)

### Advanced RSRS Analysis

```bash
python readability_rsrs.py <input_directory> [--output report.csv] [--rsrs] [--model model_name]
```

Arguments:
- `input_directory`: Directory containing markdown files to analyze
- `--output`: Optional output CSV file name (default: readability_report.csv)
- `--rsrs`: Include RSRS calculation (uses transformer models)
- `--model`: Language model to use for RSRS calculation (default: roberta-base)

### DOCX to Markdown Conversion

```bash
python docx_to_md.py <input_directory> <output_directory>
```

Arguments:
- `input_directory`: Directory containing .docx files
- `output_directory`: Directory to save converted .md files

## Example Workflow

For a complete workflow from Word documents to readability analysis:

1. Convert Word documents to markdown:
   ```
   python docx_to_md.py ./docs ./markdown_docs
   ```

2. Run basic readability analysis:
   ```
   python readability_textstat.py ./markdown_docs --output basic_metrics.csv
   ```

3. Run advanced RSRS analysis:
   ```
   python readability_rsrs.py ./markdown_docs --output advanced_metrics.csv --rsrs
   ```

## Output Format

The analysis tools generate CSV files with the following information:

- Document-specific information (filename, paragraph number)
- Multiple readability metrics
- Word and sentence statistics
- RSRS scores (if requested)
- Full paragraph text

A summary statistics file is also generated with aggregate metrics.

## Dependencies

- `textstat`: For traditional readability metrics
- `beautifulsoup4`: For HTML parsing in markdown files
- `pandas`: For data manipulation and CSV output
- `transformers` and `torch`: For RSRS calculation
- `nltk`: For sentence tokenization
- `docx2python`: For DOCX conversion
- `tqdm`: For progress bars

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
