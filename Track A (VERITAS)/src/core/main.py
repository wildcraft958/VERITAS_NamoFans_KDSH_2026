"""
KDSH 2026: Constraint-Driven Narrative Auditor

Main entry point for the 5-layer pipeline.

Usage:
    python -m core.main --backstory <file> --novel <file> [--output <file>]
    python -m core.main --batch <csv> --novels-dir <dir> [--output <csv>]
    python -m core.main --serve
"""

import argparse
import json
import sys
from pathlib import Path

from core.workflow import run_pipeline
from core.api.server import process_batch, run_server


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KDSH 2026: Constraint-Driven Narrative Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python -m core.main --backstory story.txt --novel novel.txt
  
  # Batch processing
  python -m core.main --batch train.csv --novels-dir Books/ --output results.csv
  
  # Start API server
  python -m core.main --serve
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true", help="Single prediction mode")
    mode.add_argument("--batch", type=str, help="Batch mode: path to CSV file")
    mode.add_argument("--serve", action="store_true", help="Start API server")
    
    # Single mode arguments
    parser.add_argument("--backstory", type=str, help="Path to backstory text file")
    parser.add_argument("--novel", type=str, help="Path to novel text file")
    
    # Batch mode arguments
    parser.add_argument("--novels-dir", type=str, help="Directory containing novel files")
    
    # Output
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.single:
        # Single prediction mode
        if not args.backstory or not args.novel:
            parser.error("--single mode requires --backstory and --novel")
        
        run_single(args.backstory, args.novel, args.output, args.verbose)
    
    elif args.batch:
        # Batch processing mode
        if not args.novels_dir:
            parser.error("--batch mode requires --novels-dir")
        
        output = args.output or "predictions.csv"
        process_batch(args.batch, args.novels_dir, output)
    
    elif args.serve:
        # API server mode
        run_server()


def run_single(
    backstory_path: str,
    novel_path: str,
    output_path: str | None,
    verbose: bool
) -> None:
    """Run single prediction."""
    # Load files
    with open(backstory_path) as f:
        backstory = f.read()
    
    with open(novel_path) as f:
        novel_text = f.read()
    
    print(f"Backstory: {len(backstory)} characters")
    print(f"Novel: {len(novel_text)} characters")
    print("Running 5-layer pipeline...")
    print()
    
    # Run pipeline
    result = run_pipeline(
        backstory=backstory,
        novel_text=novel_text,
        backstory_id=Path(backstory_path).stem,
        novel_id=Path(novel_path).stem
    )
    
    # Output result
    print("=" * 60)
    print(f"PREDICTION: {result['prediction']}")
    print(f"VERDICT: {'CONSISTENT' if result['prediction'] == 1 else 'CONTRADICTORY'}")
    print(f"CONFIDENCE: {result['confidence']:.2f}")
    print()
    
    if result.get("reasoning"):
        print("REASONING:")
        for i, reason in enumerate(result["reasoning"], 1):
            print(f"  {i}. {reason}")
    
    if result.get("error"):
        print(f"\nERROR: {result['error']}")
    
    # Save to file if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull result saved to: {output_path}")


if __name__ == "__main__":
    main()
