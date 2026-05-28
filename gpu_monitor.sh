#!/bin/bash

# Continuously monitor GPU utilization with dynamic terminal resizing
while true; do
    # Capture nvidia-smi output
    OUTPUT=$(nvidia-smi)

    # Count rows and find max line length
    ROWS=$(echo "$OUTPUT" | wc -l)
    COLS=$(echo "$OUTPUT" | awk '{print length}' | sort -nr | head -1)

    # Resize terminal dynamically
    printf '\036[8;%d;%dt' "$ROWS" "$COLS"

    # Clear screen and print fresh output
    clear
    echo "$OUTPUT"

    # Wait 1 second
    sleep 1
done

