import pandas as pd
import os
import glob
from pathlib import Path
import argparse

def sum_bitslice_csv_files(output_dir="output/bitslice", pattern="class*/bitslice/bitslice_analysis.csv", discard_overflow=False):
    """
    Read bitslice CSV files and sum total_bitslice_amount and overflow_count for each file
    
    Args:
        output_dir: Directory containing class folders (relative to project root)
        pattern: Glob pattern to find CSV files relative to output_dir
    """
    base_path = Path.cwd() / output_dir
    csv_files = glob.glob(str(base_path / pattern))
    csv_files.sort()
    
    if not csv_files:
        print(f"No CSV files found in {base_path / pattern}")
        return []
    
    print(f"Found {len(csv_files)} CSV files")
    print("-" * 100)
    
    results = []
    total_bitslices = 0
    total_overflows = 0
    total_effective = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Sum both bitslices and overflows
            file_bitslices = df['total_bitslice_amount'].sum()
            file_overflows = df['overflow_count'].sum()
            if discard_overflow:
                effective_bitslices = file_bitslices + file_overflows
            else:
                effective_bitslices = file_bitslices - file_overflows
            
            class_name = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
            
            results.append({
                'class': class_name,
                'file': csv_file,
                'total_bitslice_sum': file_bitslices,
                'total_overflow_sum': file_overflows,
                'effective_bitslices': effective_bitslices,
                'num_rows': len(df)
            })
            
            total_bitslices += file_bitslices
            total_overflows += file_overflows
            total_effective += effective_bitslices
            
            if discard_overflow:
                print(f"{class_name}: {file_bitslices:,} bitslices + {file_overflows:,} overflows = {effective_bitslices:,} bitslices ({len(df)} rows)")
            else:
                print(f"{class_name}: {file_bitslices:,} bitslices - {file_overflows:,} overflows = {effective_bitslices:,} bitslices ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    print("-" * 100)
    if discard_overflow:
        print(f"TOTAL: {total_bitslices:,} bitslices + {total_overflows:,} overflows = {total_effective:,} bitslices")
    else:
        print(f"TOTAL: {total_bitslices:,} bitslices - {total_overflows:,} overflows = {total_effective:,} bitslices")
    
    return results

def save_summary(results, output_dir="output/summary", output_file="bitslice_summary.csv"):
    """Save summary results to CSV in output/summary/ directory"""
    if not results:
        print("No results to save")
        return
    
    summary_df = pd.DataFrame(results)
    
    # Create output/summary/ directory structure
    summary_dir = Path.cwd() / output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = summary_dir / output_file
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to {output_path}")
    return summary_df

def print_statistics(results):
    """Print basic statistics for bitslices, overflows, and effective bitslices"""
    if not results:
        return
    
    bitslice_sums = [r['total_bitslice_sum'] for r in results]
    overflow_sums = [r['total_overflow_sum'] for r in results]
    effective_sums = [r['effective_bitslices'] for r in results]
    
    print(f"\nBitslice Statistics:")
    print(f"Classes analyzed: {len(results)}")
    print(f"Average per class: {sum(bitslice_sums) / len(bitslice_sums):,.0f}")
    print(f"Min: {min(bitslice_sums):,}")
    print(f"Max: {max(bitslice_sums):,}")
    
    print(f"\nOverflow Statistics:")
    print(f"Average per class: {sum(overflow_sums) / len(overflow_sums):,.0f}")
    print(f"Min: {min(overflow_sums):,}")
    print(f"Max: {max(overflow_sums):,}")
    
    print(f"\nEffective Bitslices (bitslices - overflows):")
    print(f"Average per class: {sum(effective_sums) / len(effective_sums):,.0f}")
    print(f"Min: {min(effective_sums):,}")
    print(f"Max: {max(effective_sums):,}")
    print(f"Overflow rate: {sum(overflow_sums) / sum(bitslice_sums) * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Summarize bitslice analysis CSV files")
    parser.add_argument("--input-dir", default="output/bitslice", help="Directory containing class folders (relative to project root)")
    parser.add_argument("--output-dir", default="output/summary", help="Directory storing summary (relative to project root)")
    parser.add_argument("--pattern", default="class*/bitslice/bitslice_analysis.csv",help="Glob pattern for CSV files")
    parser.add_argument("--save", action="store_true", help="Save summary to CSV file")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--bitslice-keep-overflow", action="store_true", default=False, help="Keep overflow slice in bitslice method")
    args = parser.parse_args()
    
    results = sum_bitslice_csv_files(args.input_dir, args.pattern, not args.bitslice_keep_overflow)
    
    if args.stats:
        print_statistics(results)
    
    if args.save:
        save_summary(results, args.output_dir)

if __name__ == "__main__":
    main()