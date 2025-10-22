#!/bin/bash

SCP_DIR="/home/work_nfs19/hkxie/modified_scp"
OUTPUT_BASE="/home/work_nfs19/hkxie/data/lance_test/token_lance"

for i in {1..8}
do
    scp_file="${SCP_DIR}/part_${i}.scp"
    if [ -f "$scp_file" ]; then
        python writer.py --scp_file "$scp_file" --output_base_dir "$OUTPUT_BASE" --buffer_size 10000 &
    else
        echo "File $scp_file not found, skipping."
    fi
done

wait
echo "All processes finished."
