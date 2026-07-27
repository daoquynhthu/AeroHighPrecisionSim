import csv
import os
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/dart/aerodynamics_table.csv"
TEMP_PATH = CSV_PATH.rsplit(".", 1)[0] + "_clean.csv"

def clean_table():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    try:
        with open(CSV_PATH, 'r', newline='') as infile, \
             open(TEMP_PATH, 'w', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader)
            writer.writerow(header)

            # Find indices
            try:
                idx_beta = header.index("Beta")
                idx_cy = header.index("CY")
                idx_cl = header.index("Cl")
                idx_cn = header.index("Cn")
            except ValueError as e:
                print(f"Error finding columns: {e}")
                return

            count = 0
            modified = 0

            for row in reader:
                beta = float(row[idx_beta])

                # If Beta is ~0, force symmetry
                if abs(beta) < 1e-6:
                    row[idx_cy] = "0.000000"
                    row[idx_cl] = "0.000000"
                    row[idx_cn] = "0.000000"
                    modified += 1

                writer.writerow(row)
                count += 1

            print(f"Processed {count} rows. Modified {modified} rows (enforced symmetry at Beta=0).")

        # Replace original file
        os.replace(TEMP_PATH, CSV_PATH)
        print(f"Successfully updated {CSV_PATH}")

    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    clean_table()
