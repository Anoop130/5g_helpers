import re
import csv
import os
import glob
import json
import pandas as pd

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

# ============ gets avg of before, during and after logs =============
def get_avg_ru():
    columns_of_interest = [
        "RX_TOTAL", "RX_ON_TIME", "RX_EARLY", "RX_LATE",
        "RX_ON_TIME_C", "RX_EARLY_C", "RX_LATE_C",
        "RX_ON_TIME_C_U", "RX_EARLY_C_U", "RX_LATE_C_U",
        "RX_CORRUPT", "RX_ERR_DROP", "TX_TOTAL"
    ]

    ru_csv_dir = "ru_csv"
    results = []

    for filename in os.listdir(ru_csv_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(ru_csv_dir, filename)
            try:
                df = pd.read_csv(file_path)
                df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

                first_5 = df.iloc[:5][columns_of_interest].mean()
                next_10 = df.iloc[5:15][columns_of_interest].mean()

                if len(df) > 15:
                    final_rows = df.iloc[15:20]
                else:
                    final_rows = pd.DataFrame(columns=columns_of_interest)

                final_avg = final_rows[columns_of_interest].mean()
                count_final = len(final_rows)

                results.append({
                    "file": filename,
                    "before": first_5.to_dict(),
                    "during": next_10.to_dict(),
                    "after": final_avg.to_dict(),
                })

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    with open("ru_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n Summary saved to: ru_summary.json")


# ========= json to csv =====================
def to_csv():

    with open("ru_summary.json") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    df.columns = [col.replace('.', '_') for col in df.columns]
    df.to_csv("ru_summary.csv", index=False)
    
    print("CSV saved to ru_summary.csv")

# ========= processing ru files =============
def process_ru():
    import pandas as pd

    # Load CSV
    df = pd.read_csv("ru_summary.csv")
    
    # Clean 'file' column
    df['file'] = df['file'].str.replace(r'^ru_', '', regex=True)
    df['file'] = df['file'].str.replace(r'\.csv$', '', regex=True)

    # Save intermediate cleaned version
    df.to_csv("ru_summary.csv", index=False)
    print("Cleaned CSV saved to ru_summary.csv")

    # Check for NaNs
    nan_summary = df.isna().sum(axis=1)
    files_with_nans = df.loc[nan_summary > 0, 'file'].tolist()

    if files_with_nans:
        with open("crash_summary.txt", "w") as f:
            for filename in files_with_nans:
                f.write(f"{filename}\n")
        print(f"\n{len(files_with_nans)} files had NaN values. Names written to crash_summary.txt")
        df = df[~df['file'].isin(files_with_nans)]

    # === Initial Metrics ===
    rx_metrics = [
        "RX_TOTAL", "RX_ON_TIME", "RX_EARLY", "RX_LATE",
        "RX_ON_TIME_C", "RX_EARLY_C", "RX_LATE_C",
        "RX_ON_TIME_C_U", "RX_EARLY_C_U", "RX_LATE_C_U",
        "RX_CORRUPT", "RX_ERR_DROP"
    ]

    # === Check if RX_CORRUPT and RX_ERR_DROP are always 0 ===
    zero_check_cols = [
        "before_RX_CORRUPT", "during_RX_CORRUPT", "after_RX_CORRUPT",
        "before_RX_ERR_DROP", "during_RX_ERR_DROP", "after_RX_ERR_DROP"
    ]

    # Keep only existing columns from the list
    existing_zero_cols = [col for col in zero_check_cols if col in df.columns]

    if existing_zero_cols:
        all_zero = (df[existing_zero_cols] == 0).all(axis=1)
        non_zero_files = df.loc[~all_zero, 'file'].tolist()

        if not non_zero_files:
            print("RX_CORRUPT and RX_ERR_DROP are always 0 — removing them from metrics.")
            rx_metrics = [m for m in rx_metrics if m not in ["RX_CORRUPT", "RX_ERR_DROP"]]
            df.drop(columns=existing_zero_cols, inplace=True)
        else:
            print("Non-zero RX_CORRUPT or RX_ERR_DROP found in the following files:")
            for file in non_zero_files:
                print(f" - {file}")
    else:
        print("ℹRX_CORRUPT and RX_ERR_DROP columns not found in data. Skipping zero check.")


    # === Percent Calculation ===
    phases = ['before', 'during', 'after']
    for phase in phases:
        for metric in rx_metrics:
            value_col = f"{phase}_{metric}"
            base_col = "before_RX_TOTAL"
            pct_col = f"pct_{phase}_{metric}"
            if value_col in df.columns and base_col in df.columns:
                df[pct_col] = 100 * df[value_col] / df[base_col]

        tx_col = f"{phase}_TX_TOTAL"
        pct_tx_col = f"pct_{phase}_TX_TOTAL"
        if tx_col in df.columns and "before_TX_TOTAL" in df.columns:
            df[pct_tx_col] = 100 * df[tx_col] / df["before_TX_TOTAL"]

    # === Delta Calculation ===
    delta_metrics = rx_metrics + ["RX_TOTAL", "TX_TOTAL"]
    for metric in delta_metrics:
        pct_before = f"pct_before_{metric}"
        pct_during = f"pct_during_{metric}"
        pct_after = f"pct_after_{metric}"
        delta_during = f"delta_during_{metric}"
        delta_after = f"delta_after_{metric}"

        if pct_before in df.columns and pct_during in df.columns:
            df[delta_during] = df[pct_during] - df[pct_before]
        if pct_before in df.columns and pct_after in df.columns:
            df[delta_after] = df[pct_after] - df[pct_before]

    # === Save Output ===
    df.to_csv("ru_summary.csv", index=False)
    print("Final summary with percentage and delta values saved to ru_summary.csv")

    return df






# ========= print ru results in terminal ===============
def display_ru():
    df = pd.read_csv("ru_summary.csv")

    ordered_metrics = [
        "RX_TOTAL", "RX_ON_TIME", "RX_EARLY", "RX_LATE",
        "RX_ON_TIME_C", "RX_EARLY_C", "RX_LATE_C",
        "RX_ON_TIME_C_U", "RX_EARLY_C_U", "RX_LATE_C_U",
        "TX_TOTAL"
    ]

    for _, row in df.iterrows():
        print(f"\n{row['file']}:\n")
        print(f"{'Metric':<25} | {'Before':>10} | {'%':>7} | {'During':>10} | {'%':>7} | {'After':>10} | {'%':>7} | {'ΔDuring':>9} | {'ΔAfter':>9}")
        print("-" * 110)

        for metric in ordered_metrics:
            b_val = row.get(f"before_{metric}", float('nan'))
            d_val = row.get(f"during_{metric}", float('nan'))
            a_val = row.get(f"after_{metric}", float('nan'))

            b_pct = row.get(f"pct_before_{metric}", float('nan'))
            d_pct = row.get(f"pct_during_{metric}", float('nan'))
            a_pct = row.get(f"pct_after_{metric}", float('nan'))

            delta_d = row.get(f"delta_during_{metric}", float('nan'))
            delta_a = row.get(f"delta_after_{metric}", float('nan'))

            print(f"{metric:<25} | {b_val:>10.2f} | {b_pct:>6.2f}% | "
                  f"{d_val:>10.2f} | {d_pct:>6.2f}% | {a_val:>10.2f} | {a_pct:>6.2f}% | "
                  f"{delta_d:>8.2f}% | {delta_a:>8.2f}%")




# === Run Batch ===
if __name__ == "__main__":

    parse_batch("attack_results")
    print("ru_emulator logs saved in ru_csv")
    print("gNB logs saved in gnb_csv")

    # get_avg_ru()
    # to_csv()
    process_ru()
    display_ru()