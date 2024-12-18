import os
import re
import shutil
from pathlib import Path

def clean_name(name):
    """Remove file extensions and clean the component name."""
    # Remove file extensions
    name = os.path.splitext(name)[0]
    # Remove any trailing/leading whitespace
    return name.strip()

def parse_filename(filename):
    """Parse the filename into its components."""
    # Remove the prefix and any file extension
    base = os.path.splitext(filename)[0].replace('probresult_', '')
    
    # Split into components
    parts = base.split('_')
    
    # Extract task name (now handling patterns like Y_bucket_num_events)
    task_parts = []
    i = 0
    while i < len(parts):
        current_part = parts[i]
        if current_part in ['Y', 'XY'] or 'way' in current_part or 'bool' in current_part or 'bucket' in current_part:
            while i < len(parts):
                if 'muc' in parts[i] or parts[i] == 'X':
                    break
                task_parts.append(parts[i])
                i += 1
            break
        i += 1
    task_name = '_'.join(task_parts)
    
    # Skip any 'X' or dataset indicators
    while i < len(parts) and (parts[i] == 'X' or 'muc' in parts[i]):
        i += 1
    
    # Extract model name
    model_parts = []
    while i < len(parts):
        if parts[i].startswith('epoch'):
            break
        if parts[i] not in ['X']:
            # Replace 'raw' with 'NaiveBERT'
            part = 'NaiveBERT' if parts[i] == 'raw' else parts[i]
            model_parts.append(part)
        i += 1
    model_name = '_'.join(model_parts)
    
    # Extract encoder epoch
    encoder_epoch = None
    while i < len(parts):
        if parts[i].startswith('epoch'):
            # Take the first epoch number we find
            if not encoder_epoch:
                encoder_epoch = parts[i]
        i += 1
    
    # Extract hidden state size
    hidden_size = None
    for part in parts:
        if part.startswith('nhid'):
            hidden_size = part
            break
    
    # Determine category based on presence of Sent/sent
    category = "SentCat" if ('Sent' in filename or 'sent' in filename) else "FullText"
    
    return {
        'task': clean_name(task_name),
        'category': category,
        'model': clean_name(model_name),
        'encoder_epoch': clean_name(encoder_epoch) if encoder_epoch else None,
        'hidden_size': clean_name(hidden_size) if hidden_size else None
    }

def organize_files(source_dir):
    """Organize files into a hierarchical folder structure."""
    # Create base output directory
    output_base = os.path.join(source_dir, 'organized_files')
    os.makedirs(output_base, exist_ok=True)
    
    processed = 0
    
    # Process each file
    for filename in os.listdir(source_dir):
        if not filename.startswith('probresult_'):
            continue
            
        try:
            components = parse_filename(filename)
            
            # Create folder structure
            folder_path = output_base
            for component in ['task', 'category', 'model', 'encoder_epoch', 'hidden_size']:
                if components[component]:
                    folder_path = os.path.join(folder_path, components[component])
                    os.makedirs(folder_path, exist_ok=True)
            
            # Move file
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(folder_path, filename)
            shutil.copy2(source_path, dest_path)  # Using copy2 to preserve metadata
            processed += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return processed

if __name__ == "__main__":
    source_directory = "."  # Current directory, modify as needed
    processed = organize_files(source_directory)
    print(f"Organization complete! Processed {processed} files")