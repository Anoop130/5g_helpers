import re
import csv
import os
import glob

def clean_ansi_escape(line):
    """Remove ANSI escape sequences like \x1b[0m."""
    return re.sub(r'\x1b\[[0-9;]*m', '', line)

def get_csv_output_path(log_path, subfolder):
    """Generate output CSV path in specified subfolder."""
    base = os.path.basename(log_path)
    csv_name = base.replace(".log", ".csv") if base.endswith(".log") else base + ".csv"
    folder = os.path.join(os.getcwd(), subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, csv_name)

# === RU Log Parser ===
def parse_ru(log_file_path):
    output_csv_path = get_csv_output_path(log_file_path, subfolder="ru_csv")

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

    # print(f"[RU] Saved: {output_csv_path}")

# === gNB Log Parser ===
def parse_gnb(log_file_path):
    output_csv_path = get_csv_output_path(log_file_path, subfolder="gnb_csv")

    with open(log_file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    headers = []
    data_rows = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("|--------------------DL"):
            if i + 1 < len(lines):
                header_line = lines[i + 1]
                headers = [h.strip() for h in header_line.replace('|', ' ').split()]
                i += 2
                continue
        if headers and "|" in line and not line.startswith("|--") and not any(k in line for k in ["DL", "UL"]):
            values = [v.strip() for v in line.split("|")]
            row = []
            for v in values:
                row.extend(v.split())
            if len(row) == len(headers):
                data_rows.append(row)
        i += 1

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

    # print(f"[gNB] Saved: {output_csv_path}")

# === Batch Parser ===
def parse_batch(base_folder="attack_results"):
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    n = 1
    for folder in subfolders:
        ru_logs = glob.glob(os.path.join(folder, "ru_*.log"))
        gnb_logs = glob.glob(os.path.join(folder, "gnb_*.log"))

        if ru_logs:
            # print(f"\n[>>] Parsing RU log: {ru_logs[0]}")
            parse_ru(ru_logs[0])
        else:
            print(f"[--]****No RU log found in**** {folder}")

        if gnb_logs:
            # print(f"[>>] Parsing gNB log: {gnb_logs[0]}")
            parse_gnb(gnb_logs[0])
        else:
            print(f"[--]****No gNB log found in**** {folder}\n")

        n +=1 

# === Run Batch ===
if __name__ == "__main__":
    parse_batch("attack_results")
    print("ru_emulator logs saved in ru_csv")
    print("gNB logs saved in gnb_csv")