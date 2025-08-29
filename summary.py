# summary.py
import glob
from collections import defaultdict

def generate_summary():
    weapon_totals = defaultdict(int)
    csv_files = glob.glob("result/*_result.csv")

    for csv_path in csv_files:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    name, count = line.strip().split(":")
                    weapon_totals[name] += int(count)

    with open("result/summary.csv", "w", encoding="utf-8") as f:
        for name, total in sorted(weapon_totals.items()):
            f.write(f"{name}:{total}\n")

    print("総カウント:", sum(weapon_totals.values()))