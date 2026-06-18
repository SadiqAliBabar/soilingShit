import os
import argparse
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser(description="Short script to subset 2 inverters from data.")
    ap.add_argument("--input", "-i", required=True, help="Input CSV or Excel file")
    ap.add_argument("--output", "-o", default="data/clean_soiling_data_short.csv", help="Output file")
    args = ap.parse_args()

    print(f"Loading data from {args.input}...")
    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)
        
    inverters = df["inverter_id"].dropna().unique()
    
    if len(inverters) > 2:
        keep_inverters = inverters[:2]
        print(f"Found {len(inverters)} inverters. Subsetting to 2: {keep_inverters}")
        df_short = df[df["inverter_id"].isin(keep_inverters)]
    else:
        print(f"Only found {len(inverters)} inverters. No need to subset.")
        df_short = df
        
    if args.output.endswith(".xlsx"):
        df_short.to_excel(args.output, index=False)
    else:
        df_short.to_csv(args.output, index=False)
        
    print(f"Saved subset data with {len(df_short)} rows to {args.output}")

if __name__ == "__main__":
    main()
