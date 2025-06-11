import os
import pandas as pd
import json

columns_of_interest = [
    "RX_TOTAL", "RX_ON_TIME", "RX_EARLY", "RX_LATE",
    "RX_ON_TIME_C", "RX_EARLY_C", "RX_LATE_C",
    "RX_ON_TIME_C_U", "RX_EARLY_C_U", "RX_LATE_C_U",
    "RX_CORRUPT", "RX_ERR_DROP", "TX_TOTAL"
]

def analyze_csv(file_path):
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

        return {
            "file": os.path.basename(file_path),
            "avg_first_5": first_5.to_dict(),
            "avg_next_10": next_10.to_dict(),
            "avg_final_5": final_avg.to_dict(),
            "count_final_5": count_final
        }

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def get_summary():
    results = []
    ru_csv_dir = "ru_csv"

    for filename in os.listdir(ru_csv_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(ru_csv_dir, filename)
            result = analyze_csv(full_path)
            if result:
                results.append(result)

    for r in results:
        df_path = os.path.join("ru_csv", r["file"])
        try:
            df = pd.read_csv(df_path)
            after_count = len(df.iloc[15:20])
        except:
            after_count = "?"

        print(f"\nFile: {r['file']}")
        print(f"{'Metric':<20} | {'Before attack (5)':>18} | {'During attack (10)':>18} | {'After attack (' + str(after_count) + ')':>18}")
        print("-" * 76)

        for key in r["avg_first_5"].keys():
            f5 = r["avg_first_5"].get(key, float('nan'))
            n10 = r["avg_next_10"].get(key, float('nan'))
            after = r["avg_final_5"].get(key, float('nan'))
            print(f"{key:<20} | {f5:18.2f} | {n10:18.2f} | {after:18.2f}")

    with open("ru_csv_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSummary saved to: ru_csv_summary.json")

if __name__ == "__main__":
    get_summary()
