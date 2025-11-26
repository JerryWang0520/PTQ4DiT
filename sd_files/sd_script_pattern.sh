#!/bin/bash
set -euo pipefail

BASE_DIR="output/coco2017/patterns"

find "$BASE_DIR" -type d -path "*/Final/active" | while read -r ACTIVE_DIR; do
    echo "=== Processing: $ACTIVE_DIR ==="

    for t in $(seq -w 02 50); do
        for b in ib01 ob11; do
            folder="t${t}_${b}_1_tb0_ff_n2"
            TARGET_DIR="$ACTIVE_DIR/$folder"

            if [ -d "$TARGET_DIR" ]; then
                TAR_FILE="$ACTIVE_DIR/${folder}.tar.gz"

                echo "Compressing $TARGET_DIR -> $TAR_FILE"
                tar -zcf "$TAR_FILE" -C "$ACTIVE_DIR" "$folder"

                echo "Removing $TARGET_DIR"
                rm -rf "$TARGET_DIR"
            else
                echo "Skip: $TARGET_DIR not found."
            fi
        done
    done

done
