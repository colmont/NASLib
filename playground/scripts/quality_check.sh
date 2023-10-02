#!/bin/bash

# Check if the directory was specified
if [ -z "$1" ]
then
    echo "Please specify a directory."
    exit 1
fi

# Initialize counters
subdirs_with_errors_file=0

# Main directory
main_directory=$1

# Loop over all subdirectories
for subdir in $(find $main_directory -type d)
do
    # Ignore the main directory
    if [ $subdir != $main_directory ]
    then
        # Check for the errors.json file
        if [ -f "$subdir/errors.json" ]
        then
            ((subdirs_with_errors_file++))
        fi
    fi
done

# Count the total number of subdirectories
total_subdirs=$(find $main_directory -mindepth 1 -maxdepth 1 -type d | wc -l)

# Calculate percentage
if [ $total_subdirs -gt 0 ]
then
    percentage=$(echo "scale=2; ($subdirs_with_errors_file/$total_subdirs)*100" | bc)
else
    percentage=0
fi

# Output result
echo "$percentage% of subdirectories contain an errors.json file."
