import re
import csv
import os
import glob
import json
import pandas as pd
import numpy as np

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
        "RX_TOTAL", "RX_LATE",
        "RX_LATE_C",
        "TX_TOTAL"
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

    print("\nSummary saved to: ru_summary.json")


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
    import numpy as np

    # Load CSV
    df = pd.read_csv("ru_summary.csv")
    
    # Clean 'file' column
    df['file'] = df['file'].str.replace(r'^ru_', '', regex=True)
    df['file'] = df['file'].str.replace(r'\.csv$', '', regex=True)

    # Check for NaNs
    nan_summary = df.isna().sum(axis=1)
    files_with_nans = df.loc[nan_summary > 0, 'file'].tolist()
    if files_with_nans:
        with open("crash_summary.txt", "w") as f:
            for filename in files_with_nans:
                f.write(f"{filename}\n")
        print(f"\n{len(files_with_nans)} files had NaN values. Names written to crash_summary.txt")
        df = df[~df['file'].isin(files_with_nans)]

    # Metrics for delta calculation
    rx_metrics = ["RX_TOTAL", "RX_LATE", "RX_LATE_C", "RX_LATE_C_U"]

    # === Percent Change Calculation ===
    delta_metrics = rx_metrics + ["TX_TOTAL"]
    for metric in delta_metrics:
        val_before = f"before_{metric}"
        val_during = f"during_{metric}"
        val_after = f"after_{metric}"
        change_during = f"change_pct_during_{metric}"
        change_after = f"change_pct_after_{metric}"

        if val_before in df.columns and val_during in df.columns:
            df[change_during] = 100 * (df[val_during] - df[val_before]) / df[val_before]
        if val_before in df.columns and val_after in df.columns:
            df[change_after] = 100 * (df[val_after] - df[val_before]) / df[val_before]

    # === Label attack outcomes ===
    def classify(row):
        # RX/TX thresholds
        RXTX_NONE = 1.0     # ±1%: no impact
        RXTX_PROB = 1.0     # >+1%/-1%: problem (overload or degradation)
        RXTX_RECOV = 1.0    # ±1%: full recovery
        RXTX_NORECOV = 5.0  # >±5%: no recovery

        # Lates thresholds (using percent change)
        # No impact: <25%
        # Very minutely: 25–50%
        # Minutely: 50–100%
        # Increase: 100–1000%
        # Significant: >1000%
        def max_late(metric):
            return max(
                row.get(f"change_pct_{metric}_RX_LATE", 0),
                row.get(f"change_pct_{metric}_RX_LATE_C", 0),
                row.get(f"change_pct_{metric}_RX_LATE_C_U", 0)
            )
        lates_during = max_late("during")
        lates_after  = max_late("after")

        rx_during = row.get("change_pct_during_RX_TOTAL", 0)
        tx_during = row.get("change_pct_during_TX_TOTAL", 0)
        rx_after  = row.get("change_pct_after_RX_TOTAL", 0)
        tx_after  = row.get("change_pct_after_TX_TOTAL", 0)

        # --- RX/TX Problem Path: ---
        if (rx_during > RXTX_PROB or tx_during > RXTX_PROB):
            # Overload: Recovery
            if abs(rx_after) <= RXTX_RECOV and abs(tx_after) <= RXTX_RECOV:
                return "overload_full_recovery"
            # Overload: No recovery
            if abs(rx_after) > RXTX_NORECOV or abs(tx_after) > RXTX_NORECOV:
                return "overload_no_recovery"
            # Else, generic overload (mid)
            return "overload_partial_recovery"

        if (rx_during < -RXTX_PROB or tx_during < -RXTX_PROB):
            # Degradation: Recovery
            if abs(rx_after) <= RXTX_RECOV and abs(tx_after) <= RXTX_RECOV:
                return "degradation_full_recovery"
            # Degradation: No recovery
            if abs(rx_after) > RXTX_NORECOV or abs(tx_after) > RXTX_NORECOV:
                return "degradation_no_recovery"
            # Else, generic degradation (mid)
            return "degradation_partial_recovery"

        # --- RX/TX No Impact Path: (now evaluate Lates bands) ---
        if abs(rx_during) <= RXTX_NONE and abs(tx_during) <= RXTX_NONE:
            # LATE NO IMPACT
            if lates_during < 25:
                return "no_impact"
            # LATE VERY MINUTELY
            if 25 <= lates_during < 50:
                if lates_after < 25:
                    return "lates_very_minute_increase_full_recovery"
                elif 25 <= lates_after < 50:
                    return "lates_very_minute_increase_no_recovery"
                elif lates_after >= 50:
                    return "lates_very_minute_increase_worsen"
            # LATE MINUTELY
            if 50 <= lates_during < 100:
                if lates_after < 25:
                    return "lates_minute_increase_full_recovery"
                elif 25 <= lates_after < 50:
                    return "lates_minute_increase_slight_recovery"
                elif 50 <= lates_after < 100:
                    return "lates_minute_increase_no_recovery"
                elif lates_after >= 100:
                    return "lates_minute_increase_worsen"
            # LATE INCREASE
            if 100 <= lates_during < 1000:
                if lates_after < 25:
                    return "lates_increase_full_recovery"
                elif 25 <= lates_after < 100:
                    return "lates_increase_slight_recovery"
                elif 100 <= lates_after < 1000:
                    return "lates_increase_no_recovery"
                elif lates_after >= 1000:
                    return "lates_increase_worsen"
            # LATE SIGNIFICANT
            if lates_during >= 1000:
                if lates_after < 25:
                    return "lates_significant_increase_full_recovery"
                elif 25 <= lates_after < 1000:
                    return "lates_significant_increase_slight_recovery"
                elif lates_after >= 1000:
                    if lates_after > lates_during:
                        return "lates_significant_increase_worsen"
                    else:
                        return "lates_significant_increase_no_recovery"

        # fallback
        return "other"

    # Clean and classify
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["attack_outcome"] = df.apply(classify, axis=1)

    rank_map = {
        "overload_full_recovery":           1,
        "overload_partial_recovery":        2,
        "overload_no_recovery":             3,
        "degradation_full_recovery":        4,
        "degradation_partial_recovery":     5,
        "degradation_no_recovery":          6,
        "no_impact":                        7,
        "lates_very_minute_increase_full_recovery": 8,
        "lates_very_minute_increase_no_recovery":   9,
        "lates_very_minute_increase_worsen":       10,
        "lates_minute_increase_full_recovery":     11,
        "lates_minute_increase_slight_recovery":   12,
        "lates_minute_increase_no_recovery":       13,
        "lates_minute_increase_worsen":            14,
        "lates_increase_full_recovery":            15,
        "lates_increase_slight_recovery":          16,
        "lates_increase_no_recovery":              17,
        "lates_increase_worsen":                   18,
        "lates_significant_increase_full_recovery":19,
        "lates_significant_increase_slight_recovery": 20,
        "lates_significant_increase_no_recovery":  21,
        "lates_significant_increase_worsen":       22,
        "other":                                   99
    }
    df["outcome_rank"] = df["attack_outcome"].map(rank_map).fillna(99).astype(int)

    # Save final
    df.to_csv("ru_summary.csv", index=False)
    print("Final summary with outcome labels saved to ru_summary.csv")
    return df



# ========= print ru results in terminal ===============
def display_ru(filter_label=None):
    import pandas as pd

    df = pd.read_csv("ru_summary.csv")

    ordered_metrics = [
        "RX_TOTAL", "RX_LATE",
        "RX_LATE_C",
        "TX_TOTAL"
    ]

    if filter_label:
        df = df[df["attack_outcome"] == filter_label]
        if df.empty:
            print(f"No entries found for label: {filter_label}")
            return

    for _, row in df.iterrows():
        rank  = row["outcome_rank"]
        label = row["attack_outcome"]

        print(f"\n{row['file']} — Rank: {rank} | Label: {label}\n")
        print(f"{'Metric':<25} | {'Before':>12} | {'During':>12} | {'After':>12} | {'Δ%During':>10} | {'Δ%After':>9}")
        print("-" * 90)

        for metric in ordered_metrics:
            b_val = row.get(f"before_{metric}", float('nan'))
            d_val = row.get(f"during_{metric}", float('nan'))
            a_val = row.get(f"after_{metric}", float('nan'))

            change_d = row.get(f"change_pct_during_{metric}", float('nan'))
            change_a = row.get(f"change_pct_after_{metric}", float('nan'))

            print(f"{metric:<25} | {b_val:>12.2f} | {d_val:>12.2f} | {a_val:>12.2f} | {change_d:>9.2f}% | {change_a:>8.2f}%")



# ==== displaying labels and ranks ========================
def display_labels():

    df = pd.read_csv("ru_summary.csv")

    if not {"file", "attack_outcome", "outcome_rank"}.issubset(df.columns):
        print("Required columns not found in ru_summary.csv.")
        return

    df_sorted = df.sort_values(by="file")

    print(f"\n{'File':<40} | {'Rank':<5} | {'Outcome Label'}")
    print("-" * 65)
    for _, row in df_sorted.iterrows():
        print(f"{row['file']:<40} | {int(row['outcome_rank']):<5} | {row['attack_outcome']}")

# ==== displaying stats for all metrics ====================
# def stats_summary():
#     import pandas as pd

#     df = pd.read_csv("ru_summary.csv")
#     metrics = ["RX_TOTAL", "RX_LATE", "RX_LATE_C", "TX_TOTAL"]
#     phases  = ["before", "during", "after"]

#     print(f"\n{'Metric':<12} | {'Min':>8} | {'Max':>8} | {'Avg':>8} | {'1Q':>8} | {'3Q':>8} | {'Median':>8}")
#     print("-" * 68)

#     for m in metrics:
#         # gather all three phases into one Series
#         cols = [f"{p}_{m}" for p in phases if f"{p}_{m}" in df.columns]
#         if not cols:
#             continue
#         # stack them into a single flat Series
#         vals = pd.concat([df[c].dropna().astype(float) for c in cols], ignore_index=True)
#         if vals.empty:
#             continue

#         mn   = vals.min()
#         mx   = vals.max()
#         avg  = vals.mean()
#         q1   = vals.quantile(0.25)
#         q3   = vals.quantile(0.75)
#         med  = vals.median()

#         print(f"{m:<12} | {mn:8.2f} | {mx:8.2f} | {avg:8.2f} | {q1:8.2f} | {q3:8.2f} | {med:8.2f}")


# ========= gNB CSV Processing =========================
def process_gnb(base_dir="gnb_csv", output_file="gnb_summary.csv"):
    # 1) find all csv files under base_dir, including subfolders
    pattern = os.path.join(base_dir, "**", "*.csv")
    all_files = glob.glob(pattern, recursive=True)

    df_list = []
    for file_path in all_files:
        df = pd.read_csv(file_path)

        # drop first 5 and last 5 rows
        if len(df) <= 10:
            continue
        df = df.iloc[5:-5].copy()

        # inject filename column
        filename = os.path.basename(file_path)
        df.insert(0, "filename", filename)

        df_list.append(df)

    if not df_list:
        print("No data frames to combine (all files too short?).")
        return

    # concatenate
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Saved combined summary: {output_file}")

    # --- Now validate brate columns ---
    # DL plane should be "390M"
    bad_dl = combined.loc[combined["brate"] != "390M", "filename"].unique().tolist()
    # UL plane should be "100M"
    bad_ul = combined.loc[combined["brate.1"] != "100M", "filename"].unique().tolist()

    if not bad_dl and not bad_ul:
        print("No change in brate values in either plane")
    else:
        if bad_dl:
            print("Change in brate values in DL plane for file(s):")
            for fn in bad_dl:
                print("  -", fn)
        if bad_ul:
            print("Change in brate values in UL plane for file(s):")
            for fn in bad_ul:
                print("  -", fn)



# === Run Batch ===
if __name__ == "__main__":

    parse_batch("attack_results")
    print("ru_emulator logs saved in ru_csv")
    print("gNB logs saved in gnb_csv")

    process_gnb()

    get_avg_ru()
    to_csv()
    process_ru()
    # display_ru()
    display_labels()
    # stats_summary()