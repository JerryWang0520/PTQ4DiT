import os
import glob
import pandas as pd
import argparse
from pathlib import Path

def sum_performance_csv_files(input_dir, pattern):
    csv_pattern = os.path.join(input_dir, pattern)
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return []
    
    print(f"Found {len(csv_files)} performance CSV files")
    print("=" * 100)
    
    results = []
    total_latency = 0
    # mem_transfer_latency = 0
    # compute_latency = 0
    
    for csv_file in csv_files:
        try:
            class_name = Path(csv_file).parts[-3]
            df = pd.read_csv(csv_file)
            
            num_modules = df['module'].nunique()
            num_steps = df['step'].nunique()
            num_rows = len(df)
            
            # avg_global_latency = df['avg_global_latency'].mean() if 'avg_global_latency' in df.columns else 0
            avg_local_latency = df['avg_local_latency'].mean() if 'avg_local_latency' in df.columns else 0
            
            # Drop step column to avoid accumulation
            df = df.drop(columns=['step'], errors='ignore')
            
            # Group by module and sum numeric columns
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            module_summary = df.groupby('module')[numeric_cols].sum().reset_index()
            
            file_total = module_summary['total_latency'].sum()
            # file_mem = module_summary['mem_transfer_latency'].sum()
            # file_compute = module_summary['compute_latency'].sum()
            total_latency += file_total
            # mem_transfer_latency += file_mem
            # compute_latency += file_compute

            
            # Sum all pair counts
            pair_cols = [col for col in df.columns if col.startswith('pair_')]
            pair_sums = {col: df[col].sum() for col in pair_cols}
            
            result = {
                'class': class_name,
                'total_latency': file_total,
                # 'mem_transfer_latency': file_mem,
                # 'compute_latency': file_compute,
                # 'avg_global_latency': avg_global_latency,
                'avg_local_latency': avg_local_latency,
                'modules': num_modules,
                'steps': num_steps
            }
            result.update(pair_sums)
            results.append(result)
            
            print(f"{class_name}: total={file_total:,} ({num_modules} modules, {num_steps} steps, {num_rows} rows)")
            # print(f"{class_name}: total={file_total:,}, mem={file_mem:,}, compute={file_compute:,} ({num_modules} modules, {num_steps} steps, {num_rows} rows)")
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    print("-" * 100)
    print(f"TOTAL: total={total_latency:,}")
    # print(f"TOTAL: total={total_latency:,}, mem={mem_transfer_latency:,}, compute={compute_latency:,}")
    
    return results

def save_summary(results, output_dir, output_file="performance_summary.csv"):
    if not results:
        print("No results to save")
        return
    
    summary_df = pd.DataFrame(results)
    
    summary_dir = Path.cwd() / output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = summary_dir / output_file
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to {output_path}")
    return summary_df

def print_statistics(results):
    if not results:
        return
    
    latencies = [r['total_latency'] for r in results]
    
    print(f"\nPerformance Statistics:")
    print(f"Classes analyzed: {len(results)}")
    print(f"Average per class: {sum(latencies) / len(latencies):,.0f}")
    print(f"Min: {min(latencies):,}")
    print(f"Max: {max(latencies):,}")

def main():
    parser = argparse.ArgumentParser(description="Summarize performance analysis CSV files")
    parser.add_argument("--input-dir", default="output/bitslice", help="Directory containing class folders")
    parser.add_argument("--pattern", default="class*/performance/performance_analysis.csv", help="Glob pattern for CSV files")
    parser.add_argument("--output-dir", default="output/summary", help="Directory for output summary")
    parser.add_argument("--save", action="store_true", help="Save summary to CSV file")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()
    
    results = sum_performance_csv_files(args.input_dir, args.pattern)
    
    if args.stats:
        print_statistics(results)
    
    if args.save:
        save_summary(results, args.output_dir)

if __name__ == "__main__":
    main()