#!/bin/bash

# Script to extract all .tar.bz2 files in a specified directory.
# Usage: ./extract_tar_bz2.sh /path/to/directory

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

echo "Searching for .tar.bz2 files in '$TARGET_DIR'..."

# Find all .tar.bz2 files in the target directory (not recursive).
FILES=( "$TARGET_DIR"/*.tar.bz2 )

# Check if any files matching the pattern exist.
if [ ! -e "${FILES[0]}" ]; then
    echo "No .tar.bz2 files found in '$TARGET_DIR'."
    exit 0
fi

for FILE in "${FILES[@]}"; do
    echo "Extracting '$FILE'..."
    # -x: extract
    # -j: bzip2 decompression
    # -f: filename
    # -C: directory to change to before extracting
    # We extract into the same directory as the file.
    tar -xjf "$FILE" -C "$TARGET_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Successfully extracted '$FILE'."
    else
        echo "Failed to extract '$FILE'."
    fi
done

echo "Extraction process completed."
