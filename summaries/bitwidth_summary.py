import pandas as pd
import argparse
import glob
import os

def sum_distribution_csv_columns(csv_files, output_file):
    all_dataframes = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = numeric_cols.drop('step', errors='ignore')
        all_dataframes.append(df[numeric_cols])
    
    if not all_dataframes:
        return None
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    column_sums = combined_df.sum()
    
    result_df = pd.DataFrame({
        'value': column_sums.index,
        'sum': column_sums.values
    })
    
    if output_file:
        result_df.to_csv(output_file, index=False)
    
    return result_df

def sum_bitwidth_csv_columns(csv_files, output_file):
    all_dataframes = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = numeric_cols.drop('step', errors='ignore')
        all_dataframes.append(df[numeric_cols])
    
    if not all_dataframes:
        return None, None
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    column_sums = combined_df.sum()
    
    # Categorize data
    zero_data   = column_sums.get('zero', 0)
    below_4bit  = sum(column_sums.get(str(i), 0) for i in range(1, 5))
    exceed_4bit = sum(column_sums.get(str(i), 0) for i in range(5, 11)) + column_sums.get('overflow', 0)
    below_5bit  = sum(column_sums.get(str(i), 0) for i in range(1, 6))
    exceed_5bit = sum(column_sums.get(str(i), 0) for i in range(6, 11)) + column_sums.get('overflow', 0)
    
    categorized_result = pd.DataFrame({
        'category': ['zero_data', 'below_4bit', 'exceed_4bit', 'below_5bit', 'exceed_5bit'],
        'sum': [zero_data, below_4bit, exceed_4bit, below_5bit, exceed_5bit]
    })
    
    result_df = pd.DataFrame({
        'bitwidth': column_sums.index,
        'sum': column_sums.values
    })
    
    if output_file:
        result_df.to_csv(output_file, index=False)
        categorized_file = output_file.replace('.csv', '_categorized.csv')
        categorized_result.to_csv(categorized_file, index=False)
    
    return result_df, categorized_result

def calculate_stats(result_df):
    total   = result_df['sum'].sum()
    max_idx = result_df['sum'].idxmax()
    
    print(f"Total count: {total:,}")
    print(f"Non-zero values: {(result_df['sum'] > 0).sum()} (total {len(result_df)} values in distribution)")
    print(f"Max count: {result_df.loc[max_idx, 'sum']:,} (value: {result_df.loc[max_idx, 'value']})")

def calculate_bitwidth_stats(categorized_df):
    zero_data   = categorized_df[categorized_df['category'] == 'zero_data'  ]['sum'].iloc[0]
    below_4bit  = categorized_df[categorized_df['category'] == 'below_4bit' ]['sum'].iloc[0]
    exceed_4bit = categorized_df[categorized_df['category'] == 'exceed_4bit']['sum'].iloc[0]
    below_5bit  = categorized_df[categorized_df['category'] == 'below_5bit' ]['sum'].iloc[0]
    exceed_5bit = categorized_df[categorized_df['category'] == 'exceed_5bit']['sum'].iloc[0]
    
    total = zero_data + below_4bit + exceed_4bit
    
    print("\nBitwidth Statistics:")
    
    # 4-bit categorization
    print("4-bit Analysis:")
    print(f"  zero_data  : {zero_data:,} ({zero_data/total*100:.2f}%)")
    print(f"  below_4bit : {below_4bit:,} ({below_4bit/total*100:.2f}%)")
    print(f"  exceed_4bit: {exceed_4bit:,} ({exceed_4bit/total*100:.2f}%)")
    
    # 5-bit categorization  
    print("5-bit Analysis:")
    print(f"  zero_data  : {zero_data:,} ({zero_data/total*100:.2f}%)")
    print(f"  below_5bit : {below_5bit:,} ({below_5bit/total*100:.2f}%)")
    print(f"  exceed_5bit: {exceed_5bit:,} ({exceed_5bit/total*100:.2f}%)")
    
    print(f"Total: {total:,}")

def main():
    parser = argparse.ArgumentParser(description='Summarize bitwidth analysis CSV files')
    parser.add_argument('--input-dir', required=True, help='Input directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--distribution-pattern', help='Pattern for int9 distribution CSV files')
    parser.add_argument('--bitwidth-pattern', help='Pattern for bitwidth CSV files')
    parser.add_argument('--save', action='store_true', help='Save results to CSV')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process distribution files
    if args.distribution_pattern:
        pattern_path = os.path.join(args.input_dir, args.distribution_pattern)
        csv_files = glob.glob(pattern_path)
        
        if csv_files:
            print(f"Processing {len(csv_files)} distribution files")
            output_file = os.path.join(args.output_dir, 'distribution_sums.csv') if args.save else None
            result_df = sum_distribution_csv_columns(csv_files, output_file)
            
            if result_df is not None:
                if args.save:
                    print(f"Distribution results saved to {output_file}")
                if args.stats:
                    calculate_stats(result_df)
        else:
            print(f"No distribution files found: {pattern_path}")
    
    # Process bitwidth files
    if args.bitwidth_pattern:
        pattern_path = os.path.join(args.input_dir, args.bitwidth_pattern)
        csv_files = glob.glob(pattern_path)
        
        if csv_files:
            print(f"Processing {len(csv_files)} bitwidth files")
            output_file = os.path.join(args.output_dir, 'bitwidth_sums.csv') if args.save else None
            result_df, categorized_df = sum_bitwidth_csv_columns(csv_files, output_file)
            
            if result_df is not None:
                if args.save:
                    print(f"Bitwidth results saved to {output_file}")
                    print(f"Categorized results saved to {output_file.replace('.csv', '_categorized.csv')}")
                if args.stats:
                    calculate_bitwidth_stats(categorized_df)
        else:
            print(f"No bitwidth files found: {pattern_path}")

if __name__ == "__main__":
    main()