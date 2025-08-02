#!/usr/bin/env python3
"""
Test Data Generation Script for Parquet Tokenization Testing

This script generates various types of Parquet files for testing the streaming
parquet tokenization functionality in the llama.cpp dataset-converter.

Requirements addressed:
- 6.1: Non-tokenized Parquet files containing raw text for tokenization testing
- 6.2: Pre-tokenized Parquet/GGUF files for output comparison
- 6.3: Datasets with various text encodings, special characters, and multilingual content
"""

import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import argparse
import logging

# Virtual environment activation
VENV_PATH = "/raid/ai/venv/bin/activate"

def activate_venv():
    """Activate the virtual environment if it exists"""
    if os.path.exists(VENV_PATH):
        print(f"Virtual environment found at {VENV_PATH}")
        # Note: In Python scripts, we can't source shell scripts directly
        # The virtual environment should be activated before running this script
        venv_python = "/raid/ai/venv/bin/python"
        if os.path.exists(venv_python) and sys.executable != venv_python:
            print(f"Warning: Script should be run with {venv_python}")
            print(f"Current Python: {sys.executable}")
    else:
        print(f"Virtual environment not found at {VENV_PATH}")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_basic_text_dataset():
    """Create a basic text-only Parquet file for tokenization testing"""
    data = {
        'id': range(1, 11),
        'text': [
            "Hello world, this is a simple test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Natural language processing enables computers to understand text.",
            "Large language models can generate human-like text.",
            "Tokenization is the process of converting text to tokens.",
            "Parquet is a columnar storage format for big data.",
            "Streaming processing allows handling large datasets efficiently.",
            "Test data generation is crucial for software validation.",
            "This is the final sentence in our basic test dataset."
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def create_multilingual_dataset():
    """Create a multilingual text dataset with various languages"""
    data = {
        'id': range(1, 16),
        'text': [
            "Hello world",  # English
            "Hola mundo",   # Spanish
            "Bonjour le monde",  # French
            "Hallo Welt",   # German
            "Ciao mondo",   # Italian
            "Olá mundo",    # Portuguese
            "Привет мир",   # Russian
            "你好世界",      # Chinese
            "こんにちは世界",  # Japanese
            "안녕하세요 세계", # Korean
            "مرحبا بالعالم",  # Arabic
            "שלום עולם",     # Hebrew
            "नमस्ते दुनिया",  # Hindi
            "สวัสดีชาวโลก",   # Thai
            "Γεια σου κόσμε"  # Greek
        ],
        'language': [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
            'ar', 'he', 'hi', 'th', 'el'
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def create_special_characters_dataset():
    """Create a dataset with special characters and edge cases"""
    data = {
        'id': range(1, 11),
        'text': [
            "Text with émojis: 😀 🚀 🌟 ❤️",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Unicode: ñáéíóú àèìòù äëïöü ç ß",
            "Math symbols: ∑ ∫ ∞ ≠ ≤ ≥ ± × ÷ √",
            "Currency: $ € £ ¥ ₹ ₽ ₿",
            "Quotes: \"double\" 'single' `backtick` 'smart' \"quotes\"",
            "Newlines and\ttabs\nare\there",
            "Very long text: " + "Lorem ipsum " * 100,
            "",  # Empty string
            "   \t\n   "  # Whitespace only
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def create_large_text_dataset(num_rows=1000):
    """Create a large dataset for performance testing"""
    np.random.seed(42)  # For reproducible results
    
    # Base sentences to combine
    sentences = [
        "The artificial intelligence revolution is transforming industries.",
        "Machine learning algorithms process vast amounts of data efficiently.",
        "Natural language processing enables human-computer interaction.",
        "Deep learning models require substantial computational resources.",
        "Data preprocessing is crucial for model performance.",
        "Tokenization converts text into numerical representations.",
        "Large language models demonstrate emergent capabilities.",
        "Transformer architectures have revolutionized NLP tasks.",
        "Training datasets must be diverse and representative.",
        "Model evaluation requires comprehensive testing methodologies."
    ]
    
    data = {
        'id': range(1, num_rows + 1),
        'text': []
    }
    
    for i in range(num_rows):
        # Create varied text lengths
        num_sentences = np.random.randint(1, 6)
        selected_sentences = np.random.choice(sentences, num_sentences, replace=True)
        text = " ".join(selected_sentences)
        data['text'].append(text)
    
    df = pd.DataFrame(data)
    return df

def create_pretokenized_dataset():
    """Create a pre-tokenized dataset for comparison testing"""
    # Simulate tokenized data (using simple word-based tokenization for demo)
    data = {
        'id': range(1, 6),
        'tokens': [
            [1, 15496, 1917, 29892, 445, 338, 263, 2560, 1243, 29889],  # "Hello world, this is a simple test."
            [450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889],  # "The quick brown fox..."
            [6189, 6509, 6509, 338, 4327, 292, 278, 1917, 29889],  # "Machine learning..."
            [18385, 4086, 9068, 28936, 23226, 304, 2274, 1426, 29889],  # "Natural language processing..."
            [8218, 479, 4086, 4733, 508, 5706, 5199, 29899, 4561, 1426, 29889]  # "Large language models..."
        ],
        'original_text': [
            "Hello world, this is a simple test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Natural language processing enables computers to understand text.",
            "Large language models can generate human-like text."
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def create_mixed_content_dataset():
    """Create a dataset with both text and pre-tokenized columns"""
    data = {
        'id': range(1, 6),
        'text': [
            "This text will be tokenized on-the-fly.",
            "Another sentence for dynamic tokenization.",
            "Mixed content testing is important.",
            "Streaming tokenization should handle this.",
            "Final test sentence for mixed content."
        ],
        'pretokenized': [
            [1, 910, 1426, 674, 367, 5993, 1891, 373, 29899, 1552, 29899, 17652, 29889],
            [7280, 10541, 363, 7343, 5993, 2133, 29889],
            [23478, 2793, 6724, 338, 4100, 29889],
            [3767, 292, 5993, 2133, 881, 4386, 445, 29889],
            [9550, 1243, 10541, 363, 12849, 2793, 29889]
        ],
        'metadata': [
            "category_a", "category_b", "category_a", "category_c", "category_b"
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def save_parquet_file(df, filepath, description=""):
    """Save DataFrame as Parquet file with proper schema"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(filepath, index=False, engine='pyarrow')
        
        # Log file info
        file_size = os.path.getsize(filepath)
        logging.info(f"Created {description}: {filepath}")
        logging.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}, Size: {file_size} bytes")
        
        return True
    except Exception as e:
        logging.error(f"Failed to save {filepath}: {e}")
        return False

def validate_parquet_file(filepath):
    """Validate that the created Parquet file can be read correctly"""
    try:
        df = pd.read_parquet(filepath)
        logging.info(f"Validation successful for {filepath}")
        logging.info(f"  Shape: {df.shape}")
        logging.info(f"  Columns: {list(df.columns)}")
        return True
    except Exception as e:
        logging.error(f"Validation failed for {filepath}: {e}")
        return False

def main():
    """Main function to generate all test datasets"""
    parser = argparse.ArgumentParser(description='Generate test Parquet files for tokenization testing')
    parser.add_argument('--output-dir', default='tools/dataset-converter/tests/test_data',
                       help='Output directory for generated files')
    parser.add_argument('--large-dataset-size', type=int, default=1000,
                       help='Number of rows for large dataset')
    parser.add_argument('--validate', action='store_true',
                       help='Validate generated files after creation')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    activate_venv()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Generating test Parquet files in: {output_dir}")
    
    # Generate datasets
    datasets = [
        (create_basic_text_dataset(), "basic_text_dataset.parquet", "Basic text dataset"),
        (create_multilingual_dataset(), "multilingual_dataset.parquet", "Multilingual dataset"),
        (create_special_characters_dataset(), "special_chars_dataset.parquet", "Special characters dataset"),
        (create_large_text_dataset(args.large_dataset_size), "large_text_dataset.parquet", "Large text dataset"),
        (create_pretokenized_dataset(), "pretokenized_dataset.parquet", "Pre-tokenized dataset"),
        (create_mixed_content_dataset(), "mixed_content_dataset.parquet", "Mixed content dataset")
    ]
    
    success_count = 0
    total_count = len(datasets)
    
    for df, filename, description in datasets:
        filepath = output_dir / filename
        if save_parquet_file(df, filepath, description):
            if args.validate:
                if validate_parquet_file(filepath):
                    success_count += 1
            else:
                success_count += 1
    
    # Summary
    logging.info(f"\nGeneration complete: {success_count}/{total_count} files created successfully")
    
    if success_count == total_count:
        logging.info("All test datasets generated successfully!")
        
        # Update README
        readme_path = output_dir / "README.md"
        update_readme(readme_path)
        
        return 0
    else:
        logging.error(f"Failed to generate {total_count - success_count} files")
        return 1

def update_readme(readme_path):
    """Update the README file with information about generated datasets"""
    readme_content = """# Test Data

This directory contains sample datasets for testing the dataset converter:

## Original Test Files
- `text_dataset.txt`: A simple text dataset
- `parquet_dataset.parquet`: A simple Parquet dataset  
- `small_dataset.gguf`: A small GGUF dataset
- `corrupted_dataset.gguf`: Corrupted GGUF file for error testing
- `corrupted_dataset.parquet`: Corrupted Parquet file for error testing

## Generated Test Files for Tokenization Testing

### Text-Only Datasets (for tokenization testing)
- `basic_text_dataset.parquet`: Simple English text samples for basic tokenization testing
- `multilingual_dataset.parquet`: Text samples in multiple languages (15 languages)
- `special_chars_dataset.parquet`: Text with special characters, emojis, and edge cases
- `large_text_dataset.parquet`: Large dataset (1000+ rows) for performance testing

### Pre-tokenized Datasets (for comparison)
- `pretokenized_dataset.parquet`: Pre-tokenized sequences for validation comparison
- `mixed_content_dataset.parquet`: Dataset with both text and pre-tokenized columns

## Usage

These files are used by the test suite to verify the functionality of the dataset converter,
particularly the new streaming Parquet tokenization features.

### Test Scenarios Covered
1. **Basic tokenization**: Simple English text processing
2. **Multilingual support**: Various languages and character sets
3. **Special characters**: Emojis, symbols, and edge cases
4. **Performance testing**: Large datasets for streaming validation
5. **Mixed content**: Files with both text and pre-tokenized data
6. **Comparison validation**: Pre-tokenized data for accuracy testing

### Generated by
`generate_test_parquet.py` - Test data generation script for Parquet tokenization testing
"""
    
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        logging.info(f"Updated README: {readme_path}")
    except Exception as e:
        logging.error(f"Failed to update README: {e}")

if __name__ == "__main__":
    sys.exit(main())