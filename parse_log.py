import re
import csv

def clean_ansi_escape(line):
    """Remove ANSI escape sequences like \x1b[0m."""
    return re.sub(r'\x1b\[[0-9;]*m', '', line)

# === RU Log Parser ===
def parse_ru_log_with_repeated_headers(log_file_path, output_csv_path):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    clean_lines = [clean_ansi_escape(line).strip() for line in lines]

    headers = []
    data_rows = []
    header_line_pattern = None

    for line in clean_lines:
        if not line.startswith("|"):
            continue

        if "TIME" in line and "TX_TOTAL" in line:
            # This is a header line (might be the first or a repeated one)
            current_header = [h.strip() for h in line.strip("|").split("|")]
            if not headers:
                headers = current_header
                header_line_pattern = line.strip("|")
            continue

        if line.strip("|") == header_line_pattern:
            continue

        values = [v.strip() for v in line.strip("|").split("|")]
        if len(values) == len(headers):
            data_rows.append(values)

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"[RU] Clean CSV saved to: {output_csv_path}")


# === gNB Log Parser ===
def parse_gnb_log(log_file_path, output_csv_path):
    """
    Parses gNB log with repeated DL/UL tables.
    Extracts headers + values into unified CSV.
    """
    with open(log_file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    headers = []
    data_rows = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("|--------------------DL"):
            # Found a new table block
            if i + 1 < len(lines):
                header_line = lines[i + 1]
                headers = [h.strip() for h in header_line.replace('|', ' ').split()]
                i += 2  # move to first data line
                continue

        if headers and "|" in line and not line.startswith("|--") and not any(k in line for k in ["DL", "UL"]):
            # Parse a data line
            values = [v.strip() for v in line.split("|")]
            # Flatten any remaining splits
            row = []
            for v in values:
                row.extend(v.split())
            if len(row) == len(headers):
                data_rows.append(row)
        i += 1

    # Save to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"[gNB] Table CSV saved to: {output_csv_path}")


# === Example Usage ===
parse_ru_log_with_repeated_headers(
    log_file_path="attack_results/cp_dl_long_1mb_DR_10_INJ_DU/ru_cp_dl_long_1mb_DR_10_DU.log",
    output_csv_path="ru_log_clean.csv"
)

parse_gnb_log(
    log_file_path="attack_results/cp_dl_long_1mb_DR_10_INJ_DU/gnb_cp_dl_long_1mb_DR_10.log",
    output_csv_path="gnb_log_clean.csv"
)

