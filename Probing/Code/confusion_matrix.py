import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrices from model results'
    )
    parser.add_argument(
        '--results-dir', 
        type=str,
        required=True,
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save confusion matrix plots (defaults to results dir)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default='',
        help='Only process files containing this string (e.g. "nhid400")'
    )
    parser.add_argument(
        '--new-only',
        action='store_true',
        help='Skip generating plots that already exist'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='DPI for saved plots'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[8, 6],
        help='Figure size in inches (width height)'
    )
    return parser.parse_args()

def process_result_file(result_path, output_path, dpi, figsize):
    """Process a single result file and generate confusion matrix plot"""
    with open(result_path) as f:
        result = json.load(f)
        
    # Extract true labels and predictions
    labels = np.array([l.index(max(l)) for l in result["val_true"]])
    preds = np.array([l.index(max(l)) for l in result["val_pred"]])
    
    # Create confusion matrix
    confusion_matrix = metrics.confusion_matrix(labels, preds)
    
    # Set up the plot
    plt.figure(figsize=figsize)
    display_labels = sorted(list(set(list(labels) + list(preds))))
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=display_labels
    )
    
    # Plot and customize
    cm_display.plot()
    title = (
        f"Val acc {result['val_acc']:.2f}, "
        f"Test acc {result['test_acc']:.2f}, "
        f"Train acc {result['train_acc']:.2f}\n"
        f"Epoch {result['actual_epoch']}"
    )
    cm_display.ax_.set_title(title)
    
    # Save plot
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return confusion_matrix

def main():
    args = parse_args()
    
    # Setup directories
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each result file
    for file in results_dir.glob('probresult*.json'):
        # Apply filter if specified
        if args.filter and args.filter not in file.name:
            continue
            
        output_path = output_dir / f"{file.stem}.png"
        
        # Skip if file exists and new_only is True
        if output_path.exists() and args.new_only:
            print(f"Skipping existing plot: {output_path}")
            continue
            
        print(f"Processing: {file}")
        try:
            confusion_matrix = process_result_file(
                file, 
                output_path,
                args.dpi,
                args.figsize
            )
            print("Confusion matrix:")
            print(confusion_matrix)
            print(f"Saved plot to: {output_path}\n")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}\n")

if __name__ == '__main__':
    main()