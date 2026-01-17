"""
Data loader for KDSH 2026 Dataset

Dataset structure:
- train.csv / test.csv with columns: id, book_name, char, caption, content, label
- Books/ directory with narrative .txt files named after book_name

The 'book_name' column maps to .txt files in Books/ directory.
The 'content' column contains the backstory claim to verify.
The 'label' column is 'consistent' or 'contradict'.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


def normalize_book_name(book_name: str) -> str:
    """
    Normalize book name for file matching.
    Handles various naming conventions.
    """
    # Remove extra whitespace
    normalized = ' '.join(book_name.split())
    return normalized


def find_book_file(book_name: str, books_dir: Path) -> Optional[Path]:
    """
    Find the .txt file for a given book name.
    Tries multiple matching strategies.
    
    Args:
        book_name: Name of the book from CSV
        books_dir: Path to Books directory
        
    Returns:
        Path to the book file, or None if not found
    """
    if not books_dir.exists():
        return None
    
    normalized = normalize_book_name(book_name)
    
    # Strategy 1: Exact match
    exact_path = books_dir / f"{normalized}.txt"
    if exact_path.exists():
        return exact_path
    
    # Strategy 2: Case-insensitive match
    for txt_file in books_dir.glob("*.txt"):
        if txt_file.stem.lower() == normalized.lower():
            return txt_file
    
    # Strategy 3: Partial match (book name in filename)
    for txt_file in books_dir.glob("*.txt"):
        if normalized.lower() in txt_file.stem.lower():
            return txt_file
        if txt_file.stem.lower() in normalized.lower():
            return txt_file
    
    # Strategy 4: Match with underscores/hyphens
    normalized_underscore = normalized.replace(' ', '_')
    normalized_hyphen = normalized.replace(' ', '-')
    for txt_file in books_dir.glob("*.txt"):
        stem_lower = txt_file.stem.lower()
        if stem_lower == normalized_underscore.lower():
            return txt_file
        if stem_lower == normalized_hyphen.lower():
            return txt_file
    
    return None


def load_narrative_from_books(book_name: str, books_dir: Path) -> str:
    """
    Load narrative text from Books directory.
    
    Args:
        book_name: Name of the book from CSV
        books_dir: Path to Books directory
        
    Returns:
        Narrative text content
        
    Raises:
        FileNotFoundError: If book file not found
    """
    file_path = find_book_file(book_name, books_dir)
    
    if file_path is None:
        # List available books for debugging
        available = [f.stem for f in books_dir.glob("*.txt")] if books_dir.exists() else []
        raise FileNotFoundError(
            f"Book not found: '{book_name}'\n"
            f"Available books: {available[:10]}{'...' if len(available) > 10 else ''}"
        )
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def load_dataset_from_path(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSVs from dataset directory.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        (train_df, test_df)
    """
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    
    return train_df, test_df


def load_dataset_from_kaggle(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSVs from Kaggle dataset.
    Alias for load_dataset_from_path.
    
    Args:
        data_dir: Path to dataset directory (e.g., /kaggle/input/your-dataset/)
        
    Returns:
        (train_df, test_df)
    """
    return load_dataset_from_path(data_dir)


def convert_label(label: str) -> int:
    """
    Convert label string to binary.
    
    Args:
        label: 'consistent' or 'contradict'
        
    Returns:
        1 for consistent, 0 for contradict
    """
    if isinstance(label, str):
        label_lower = label.strip().lower()
        if label_lower in ['consistent', 'true', '1', 'yes']:
            return 1
        elif label_lower in ['contradict', 'contradiction', 'false', '0', 'no']:
            return 0
    # If already numeric
    try:
        return int(label)
    except (ValueError, TypeError):
        return 0


def prepare_training_data(
    train_df: pd.DataFrame,
    books_dir: Path,
    verbose: bool = True
) -> Tuple[List[str], List[str], List[int]]:
    """
    Prepare training data from CSV.
    
    Expected columns:
    - id: unique identifier
    - book_name: name of the book (maps to Books/{book_name}.txt)
    - char: character name
    - caption: optional section caption
    - content: backstory claim text
    - label: 'consistent' or 'contradict'
    
    Args:
        train_df: Training dataframe
        books_dir: Path to Books directory
        verbose: Print progress
        
    Returns:
        (narratives, backstories, labels)
    """
    narratives = []
    backstories = []
    labels = []
    
    # Cache loaded narratives to avoid re-reading
    narrative_cache = {}
    
    errors = []
    
    for idx, row in train_df.iterrows():
        try:
            book_name = str(row.get('book_name', '')).strip()
            
            # Get narrative from cache or load
            if book_name not in narrative_cache:
                narrative = load_narrative_from_books(book_name, books_dir)
                narrative_cache[book_name] = narrative
            else:
                narrative = narrative_cache[book_name]
            
            # Construct backstory with context
            char_name = str(row.get('char', '')).strip()
            caption = str(row.get('caption', '')).strip()
            content = str(row.get('content', '')).strip()
            
            # Build backstory claim
            if char_name and caption:
                backstory = f"Character: {char_name}\nSection: {caption}\n{content}"
            elif char_name:
                backstory = f"Character: {char_name}\n{content}"
            else:
                backstory = content
            
            # Get label
            label = convert_label(row.get('label', 0))
            
            narratives.append(narrative)
            backstories.append(backstory)
            labels.append(label)
            
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
            continue
    
    if verbose:
        print(f"Loaded {len(narratives)} training examples")
        print(f"Unique books: {len(narrative_cache)}")
        print(f"Label distribution: 1={labels.count(1)}, 0={labels.count(0)}")
        if errors:
            print(f"Errors ({len(errors)}): {errors[:3]}{'...' if len(errors) > 3 else ''}")
    
    return narratives, backstories, labels


def prepare_test_data(
    test_df: pd.DataFrame,
    books_dir: Path,
    verbose: bool = True
) -> List[Dict]:
    """
    Prepare test data from CSV.
    
    Args:
        test_df: Test dataframe
        books_dir: Path to Books directory
        verbose: Print progress
        
    Returns:
        List of dictionaries with 'id', 'narrative', 'backstory', 'char', 'book_name'
    """
    test_data = []
    narrative_cache = {}
    errors = []
    
    for idx, row in test_df.iterrows():
        try:
            sample_id = row.get('id', idx)
            book_name = str(row.get('book_name', '')).strip()
            
            # Get narrative from cache or load
            if book_name not in narrative_cache:
                narrative = load_narrative_from_books(book_name, books_dir)
                narrative_cache[book_name] = narrative
            else:
                narrative = narrative_cache[book_name]
            
            # Construct backstory
            char_name = str(row.get('char', '')).strip()
            caption = str(row.get('caption', '')).strip()
            content = str(row.get('content', '')).strip()
            
            if char_name and caption:
                backstory = f"Character: {char_name}\nSection: {caption}\n{content}"
            elif char_name:
                backstory = f"Character: {char_name}\n{content}"
            else:
                backstory = content
            
            test_data.append({
                'id': sample_id,
                'book_name': book_name,
                'char': char_name,
                'narrative': narrative,
                'backstory': backstory
            })
            
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
            # Add placeholder for failed rows
            test_data.append({
                'id': row.get('id', idx),
                'book_name': row.get('book_name', ''),
                'char': row.get('char', ''),
                'narrative': '',
                'backstory': str(row.get('content', '')),
                'error': str(e)
            })
    
    if verbose:
        print(f"Loaded {len(test_data)} test examples")
        print(f"Unique books: {len(narrative_cache)}")
        if errors:
            print(f"Errors ({len(errors)}): {errors[:3]}{'...' if len(errors) > 3 else ''}")
    
    return test_data


def get_dataset_stats(data_dir: Path, books_dir: Path) -> Dict:
    """
    Get statistics about the dataset.
    
    Args:
        data_dir: Path to CSV files
        books_dir: Path to Books directory
        
    Returns:
        Dictionary with dataset statistics
    """
    train_df, test_df = load_dataset_from_path(data_dir)
    
    # Count books in directory
    book_files = list(books_dir.glob("*.txt")) if books_dir.exists() else []
    
    # Get unique books in CSVs
    train_books = set(train_df['book_name'].unique()) if 'book_name' in train_df.columns else set()
    test_books = set(test_df['book_name'].unique()) if 'book_name' in test_df.columns else set()
    
    # Check which books are available
    available_books = {f.stem for f in book_files}
    
    stats = {
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'book_files_count': len(book_files),
        'train_unique_books': len(train_books),
        'test_unique_books': len(test_books),
        'books_in_directory': list(available_books)[:10],
        'missing_train_books': list(train_books - available_books)[:5],
        'missing_test_books': list(test_books - available_books)[:5],
    }
    
    if 'label' in train_df.columns:
        stats['label_distribution'] = train_df['label'].value_counts().to_dict()
    
    return stats


# Quick test function
def test_data_loader(data_dir: str, books_dir: str = None):
    """
    Test the data loader with your dataset.
    
    Args:
        data_dir: Path to directory with train.csv and test.csv
        books_dir: Path to Books directory (defaults to data_dir/Books)
    """
    data_path = Path(data_dir)
    books_path = Path(books_dir) if books_dir else data_path / "Books"
    
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    stats = get_dataset_stats(data_path, books_path)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Loading Training Data")
    print("=" * 60)
    train_df, test_df = load_dataset_from_path(data_path)
    
    if len(train_df) > 0:
        narratives, backstories, labels = prepare_training_data(
            train_df.head(5), books_path, verbose=True
        )
        print(f"\nSample backstory:\n{backstories[0][:200]}...")
        print(f"\nSample narrative length: {len(narratives[0]):,} characters")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_data_loader(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("Usage: python data_loader.py <data_dir> [books_dir]")
