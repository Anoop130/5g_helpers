#!/bin/bash

# === Setup ===

# Graceful exit on Ctrl+C
cleanup() {
  echo -e "\n\nScript interrupted. Cleaning up and exiting..."
  exit 1
}
trap cleanup INT

# Paths
PCAP_GEN="python3 $HOME/Dos-attacker/python_code/pcap_gen.py"
INPUT_DIR="$HOME/Dos-attacker/Traffic"
OUTPUT_DIR="$HOME/Dos-attacker/Traffic/custom"

# MAC addresses
RU_MAC="90:e3:ba:00:12:22"
DU_MAC="90:e2:ba:8e:39:51"

# VLAN ID
VLAN_ID=33

# PCAP files to process
filenames=(
    cp_dl_long_10mb.pcap
    cp_dl_long_1mb.pcap
    cp_dl_short_10mb.pcap
    cp_dl_short_1mb.pcap
    cp_ul_long_10mb.pcap
    cp_ul_long_1mb.pcap
    cp_ul_short_10mb.pcap
    cp_ul_short_1mb.pcap
    up_dl_short_10mb.pcap
    up_dl_short_1mb.pcap
    up_ul_long_10mb.pcap
    up_ul_long_1mb.pcap
    up_ul_short_10mb.pcap
    up_ul_short_1mb.pcap
)

# Rates to use
rates=(10 100 1000)

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting PCAP Generation ==="
echo "Input dir: $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# === Processing ===

expected_files=()

for file in "${filenames[@]}"; do
  for rate in "${rates[@]}"; do

    # RU → DU
    output1="${file%.pcap}_RD_${rate}.pcap"
    echo "Generating $output1 | File: $file | Direction: RU->DU | Src: $DU_MAC | Dst: $RU_MAC | Rate: ${rate} Mbps"
    if $PCAP_GEN --pcap "${INPUT_DIR}/${file}:1" \
                 --out "${OUTPUT_DIR}/${output1}" \
                 --vlan $VLAN_ID \
                 --dst $RU_MAC \
                 --src $DU_MAC \
                 --mbps $rate > /dev/null 2>> error.log; then
      echo "Success."
    else
      echo "Failed."
    fi
    expected_files+=("${OUTPUT_DIR}/${output1}")

    # DU → RU
    output2="${file%.pcap}_DR_${rate}.pcap"
    echo "Generating $output2 | File: $file | Direction: DU->RU | Src: $RU_MAC | Dst: $DU_MAC | Rate: ${rate} Mbps"
    if $PCAP_GEN --pcap "${INPUT_DIR}/${file}:1" \
                 --out "${OUTPUT_DIR}/${output2}" \
                 --vlan $VLAN_ID \
                 --dst $DU_MAC \
                 --src $RU_MAC \
                 --mbps $rate > /dev/null 2>> error.log; then
      echo "Success."
    else
      echo "Failed."
    fi
    expected_files+=("${OUTPUT_DIR}/${output2}")

  done
done

echo ""

# === Validation ===

all_present=true
for file in "${expected_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing: $file"
    all_present=false
  fi
done

if $all_present; then
  echo "All PCAPs generated and saved to: $OUTPUT_DIR"
else
  echo "Some PCAP files are missing. Check the errors above or see error.log"
fi
