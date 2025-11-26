"""
Statistical comparison of two image editing methods.

Usage:
    # Compare two CSV files (same method name, different runs/configs):
    python compare_methods.py --csv1 evaluation_result.csv --csv2 evaluation_result_original_p2p.csv
    
    # Compare two methods within the same CSV file:
    python compare_methods.py --csv1 evaluation_result.csv --method1 "1_directinversion+p2p" --method2 "1_ddim+p2p"
    
    # Specify output file for results:
    python compare_methods.py --csv1 file1.csv --csv2 file2.csv --output comparison_results.txt
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Try to import tabulate, fallback to simple formatting
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    
    def tabulate(data, headers, tablefmt=None):
        """Simple fallback table formatter."""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in col_widths)
        
        # Format rows
        rows = []
        for row in data:
            rows.append(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
        
        return "\n".join([header_line, separator] + rows)

warnings.filterwarnings('ignore')

# Define metric properties: name -> (higher_is_better, description)
METRIC_INFO = {
    "structure_distance": (False, "Structural distance (lower=better)"),
    "psnr": (True, "PSNR - full image (higher=better)"),
    "psnr_unedit_part": (True, "PSNR - background preservation (higher=better)"),
    "psnr_edit_part": (True, "PSNR - edited region (higher=better)"),
    "lpips": (False, "LPIPS - full image (lower=better)"),
    "lpips_unedit_part": (False, "LPIPS - background preservation (lower=better)"),
    "lpips_edit_part": (False, "LPIPS - edited region (lower=better)"),
    "mse": (False, "MSE - full image (lower=better)"),
    "mse_unedit_part": (False, "MSE - background preservation (lower=better)"),
    "mse_edit_part": (False, "MSE - edited region (lower=better)"),
    "ssim": (True, "SSIM - full image (higher=better)"),
    "ssim_unedit_part": (True, "SSIM - background preservation (higher=better)"),
    "ssim_edit_part": (True, "SSIM - edited region (higher=better)"),
    "clip_similarity_source_image": (True, "CLIP sim - source image (higher=better)"),
    "clip_similarity_target_image": (True, "CLIP sim - target image (higher=better)"),
    "clip_similarity_target_image_edit_part": (True, "CLIP sim - edited region (higher=better)"),
}


def load_and_extract(csv_path, method_name=None):
    """Load CSV and extract metrics for a specific method or all methods."""
    df = pd.read_csv(csv_path)
    print(df.columns)
    print(df.head())
    
    # Extract metrics for this method
    metrics_data = {'file_id': df['file_id']}
    
    for col in df.columns:
        if '|' in col:
            metric_name = col.split('|')[1]
        else:
            metric_name = col
        metrics_data[metric_name] = df[col]
    
    return pd.DataFrame(metrics_data), method_name


def compute_statistics(values):
    """Compute descriptive statistics for a series of values."""
    valid = pd.to_numeric(values, errors='coerce').dropna()
    if len(valid) == 0:
        return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'n': 0}
    return {
        'mean': valid.mean(),
        'std': valid.std(),
        'median': valid.median(),
        'min': valid.min(),
        'max': valid.max(),
        'n': len(valid)
    }


def paired_comparison(values1, values2, metric_name):
    """
    Perform paired statistical comparison between two methods.
    Returns dict with test results.
    """
    # Convert to numeric, coercing errors to NaN
    v1 = pd.to_numeric(values1, errors='coerce')
    v2 = pd.to_numeric(values2, errors='coerce')
    
    # Get paired valid samples (both must be valid)
    valid_mask = v1.notna() & v2.notna()
    v1_valid = v1[valid_mask].values
    v2_valid = v2[valid_mask].values
    
    n_paired = len(v1_valid)
    
    if n_paired < 3:
        return {
            'n_paired': n_paired,
            't_stat': np.nan, 't_pvalue': np.nan,
            'wilcoxon_stat': np.nan, 'wilcoxon_pvalue': np.nan,
            'wins_1': 0, 'wins_2': 0, 'ties': 0,
            'mean_diff': np.nan, 'effect_size': np.nan
        }
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(v1_valid, v2_valid)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(v1_valid, v2_valid)
    except ValueError:
        # All differences are zero
        wilcoxon_stat, wilcoxon_pvalue = np.nan, 1.0
    
    # Win/loss/tie counts
    higher_is_better = METRIC_INFO.get(metric_name, (True, ""))[0]
    
    if higher_is_better:
        wins_1 = np.sum(v1_valid > v2_valid)
        wins_2 = np.sum(v2_valid > v1_valid)
    else:
        wins_1 = np.sum(v1_valid < v2_valid)  # Lower is better for method 1
        wins_2 = np.sum(v2_valid < v1_valid)
    
    ties = n_paired - wins_1 - wins_2
    
    # Mean difference and effect size (Cohen's d)
    diff = v1_valid - v2_valid
    mean_diff = np.mean(diff)
    pooled_std = np.sqrt((np.var(v1_valid) + np.var(v2_valid)) / 2)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else np.nan
    
    return {
        'n_paired': n_paired,
        't_stat': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_pvalue': wilcoxon_pvalue,
        'wins_1': wins_1,
        'wins_2': wins_2,
        'ties': ties,
        'mean_diff': mean_diff,
        'effect_size': effect_size
    }


def significance_stars(p_value):
    """Return significance stars based on p-value."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def compare_methods(csv1, csv2=None, method1=None, method2=None, output=None):
    """
    Main comparison function.
    
    If csv2 is provided: compare same method across two files
    If method2 is provided: compare two methods within csv1
    """
    
    # Load data
    df1, name1 = load_and_extract(csv1, method1)
    
    if csv2 is not None:
        df2, name2 = load_and_extract(csv2, method2)
        label1 = f"{name1} (file1)"
        label2 = f"{name2} (file2)"
    elif method2 is not None:
        df2, name2 = load_and_extract(csv1, method2)
        label1 = name1
        label2 = name2
    else:
        raise ValueError("Must provide either csv2 or method2 for comparison")
    
    # Merge on file_id to ensure paired comparison
    merged = pd.merge(df1, df2, on='file_id', suffixes=('_1', '_2'))
    
    # Get common metrics
    metrics1 = set(df1.columns) - {'file_id'}
    metrics2 = set(df2.columns) - {'file_id'}
    common_metrics = sorted(metrics1 & metrics2)
    
    # Prepare results
    results = []
    summary_table = []
    
    print("=" * 80)
    print("STATISTICAL COMPARISON OF IMAGE EDITING METHODS")
    print("=" * 80)
    print(f"\nMethod 1: {label1}")
    print(f"Method 2: {label2}")
    print(f"Total paired samples: {len(merged)}")
    print()
    
    for metric in common_metrics:
        col1 = f"{metric}_1"
        col2 = f"{metric}_2"
        
        if col1 not in merged.columns or col2 not in merged.columns:
            continue
        
        stats1 = compute_statistics(merged[col1])
        stats2 = compute_statistics(merged[col2])
        comparison = paired_comparison(merged[col1], merged[col2], metric)
        
        higher_is_better = METRIC_INFO.get(metric, (True, ""))[0]
        direction = "↑" if higher_is_better else "↓"
        
        # Determine winner
        if comparison['t_pvalue'] < 0.05:
            if higher_is_better:
                winner = "Method 1" if comparison['mean_diff'] > 0 else "Method 2"
            else:
                winner = "Method 1" if comparison['mean_diff'] < 0 else "Method 2"
        else:
            winner = "No sig. diff."
        
        results.append({
            'metric': metric,
            'stats1': stats1,
            'stats2': stats2,
            'comparison': comparison,
            'winner': winner
        })
        
        # Build summary row
        sig = significance_stars(comparison['t_pvalue'])
        win_rate_1 = comparison['wins_1'] / comparison['n_paired'] * 100 if comparison['n_paired'] > 0 else 0
        
        summary_table.append([
            f"{metric} {direction}",
            f"{stats1['mean']:.4f} ± {stats1['std']:.4f}",
            f"{stats2['mean']:.4f} ± {stats2['std']:.4f}",
            f"{comparison['mean_diff']:+.4f}",
            f"{comparison['t_pvalue']:.4f}{sig}",
            f"{comparison['wins_1']}/{comparison['wins_2']}/{comparison['ties']}",
            f"{win_rate_1:.1f}%",
            winner
        ])
    
    # Print summary table
    headers = ["Metric", "Method 1", "Method 2", "Diff", "p-value", "W1/W2/T", "Win%", "Winner"]
    print(tabulate(summary_table, headers=headers, tablefmt="grid"))
    
    print("\n" + "=" * 80)
    print("LEGEND")
    print("=" * 80)
    print("↑ = Higher is better, ↓ = Lower is better")
    print("Diff = Mean(Method1) - Mean(Method2)")
    print("W1/W2/T = Wins for Method1 / Wins for Method2 / Ties")
    print("Win% = Percentage of images where Method 1 is better")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    
    # Detailed per-metric analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    for r in results:
        metric = r['metric']
        s1, s2 = r['stats1'], r['stats2']
        c = r['comparison']
        
        desc = METRIC_INFO.get(metric, (True, metric))[1]
        
        print(f"\n{metric}")
        print(f"  Description: {desc}")
        print(f"  Method 1: mean={s1['mean']:.4f}, std={s1['std']:.4f}, median={s1['median']:.4f}")
        print(f"  Method 2: mean={s2['mean']:.4f}, std={s2['std']:.4f}, median={s2['median']:.4f}")
        print(f"  Paired samples: {c['n_paired']}")
        print(f"  Mean difference: {c['mean_diff']:+.4f}")
        print(f"  Effect size (Cohen's d): {c['effect_size']:.4f}")
        print(f"  Paired t-test: t={c['t_stat']:.4f}, p={c['t_pvalue']:.4f}{significance_stars(c['t_pvalue'])}")
        print(f"  Wilcoxon test: W={c['wilcoxon_stat']:.4f}, p={c['wilcoxon_pvalue']:.4f}{significance_stars(c['wilcoxon_pvalue'])}")
        print(f"  Win rate: Method1={c['wins_1']}, Method2={c['wins_2']}, Ties={c['ties']}")
        print(f"  Conclusion: {r['winner']}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    method1_wins = sum(1 for r in results if r['winner'] == "Method 1")
    method2_wins = sum(1 for r in results if r['winner'] == "Method 2")
    no_diff = sum(1 for r in results if r['winner'] == "No sig. diff.")
    
    print(f"Metrics where Method 1 is significantly better: {method1_wins}")
    print(f"Metrics where Method 2 is significantly better: {method2_wins}")
    print(f"Metrics with no significant difference: {no_diff}")
    
    # Save to file if requested
    if output:
        import sys
        from io import StringIO
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Re-run print statements (simplified version for file)
        print("=" * 80)
        print("STATISTICAL COMPARISON OF IMAGE EDITING METHODS")
        print("=" * 80)
        print(f"\nMethod 1: {label1}")
        print(f"Method 2: {label2}")
        print(f"Total paired samples: {len(merged)}")
        print()
        print(tabulate(summary_table, headers=headers, tablefmt="grid"))
        print(f"\nOverall: Method1={method1_wins} wins, Method2={method2_wins} wins, No diff={no_diff}")
        
        output_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(output, 'w') as f:
            f.write(output_str)
        print(f"\nResults saved to: {output}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare two image editing methods statistically")
    parser.add_argument('--csv1', type=str, required=True, 
                        help="Path to first CSV file with evaluation results")
    parser.add_argument('--csv2', type=str, default=None,
                        help="Path to second CSV file (for comparing same method across files)")
    parser.add_argument('--method1', type=str, default=None,
                        help="Name of first method (auto-detected if not provided)")
    parser.add_argument('--method2', type=str, default=None,
                        help="Name of second method (for comparing methods within same file)")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save comparison results")
    
    args = parser.parse_args()
    
    if args.csv2 is None and args.method2 is None:
        parser.error("Must provide either --csv2 or --method2 for comparison")
    
    compare_methods(args.csv1, args.csv2, args.method1, args.method2, args.output)


if __name__ == "__main__":
    main()
