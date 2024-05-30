#!/bin/bash

# Check if the path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 path_to_compressed_files"
  exit 1
fi

# Change to the specified directory
cd "$1" || exit 1

# Loop through all compressed files in the specified directory
for file in *.{zip,rar,tgz}; do
  # Check if the file exists to avoid errors when no files match
  if [ -e "$file" ]; then
    # Get the base name of the file without the extension
    base_name=$(basename "$file" .zip)
    base_name=$(basename "$base_name" .rar)
    base_name=$(basename "$base_name" .tgz)

    # Create a directory with the base name
    mkdir -p "$base_name"

    # Check the file extension and extract accordingly
    case "$file" in
      *.zip)
        echo "Extracting $file to $base_name..."
        unzip "$file" -d "$base_name"
        ;;
      *.rar)
        echo "Extracting $file to $base_name..."
        unrar x "$file" "$base_name/"
        ;;
      *.tgz)
        echo "Extracting $file to $base_name..."
        tar -xzf "$file" -C "$base_name"
        ;;
    esac
  fi
done

echo "Extraction complete."
