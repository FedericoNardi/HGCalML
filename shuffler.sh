#!/bin/bash

# Set the folder path
folder_path="/media/disk/photon_data/v2/unif"

# Find all files in the folder and its subfolders
file_list=$(find "$folder_path" -type f)

# Convert file list to an array
IFS=$'\n' read -d '' -ra files <<< "$file_list"

# Shuffle the array using Fisher-Yates algorithm
for ((i = ${#files[@]} - 1; i > 0; i--)); do
    j=$((RANDOM % (i + 1)))
    temp="${files[i]}"
    files[i]="${files[j]}"
    files[j]="$temp"
done

# Calculate the number of files for each output file
total_files=${#files[@]}
twentyfive_percent=$((total_files / 4))
seventyfive_percent=$((total_files - twentyfive_percent))

# Create output files
printf "%s\n" "${files[@]:0:twentyfive_percent}" > "photons_test.txt"
printf "%s\n" "${files[@]:twentyfive_percent:seventyfive_percent}" > "photons_train.txt"

echo "Files shuffled successfully and saved in output_25percent.txt and output_75percent.txt."
