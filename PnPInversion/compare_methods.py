"""
Statistical comparison of image editing methods.

Supports side-by-side tables and LaTeX output for comparing multiple methods.

Usage:
    # Compare multiple CSV files (methods auto-detected):
    python compare_methods.py --csvs file1.csv file2.csv file3.csv
    
    # Compare multiple CSV files with specific methods:
    python compare_methods.py --csvs file1.csv file2.csv --methods method1 method2
    
    # Compare two methods within the same CSV file:
    python compare_methods.py --csvs file.csv file.csv --methods "1_directinversion+p2p" "1_ddim+p2p"
    
    # Specify output file for results:
    python compare_methods.py --csvs file1.csv file2.csv --output comparison_results.txt
    
    # Generate LaTeX table:
    python compare_methods.py --csvs file1.csv file2.csv file3.csv --latex comparison_table.tex
    python compare_methods.py --csvs file.csv file.csv --methods method1 method2 --latex comparison_table.tex
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
    
    # Extract metrics for this method
    metrics_data = {'file_id': df['file_id']}
    
    for col in df.columns:
        if '|' in col:
            method_part, metric_name = col.split('|', 1)
            # If method_name is specified, only include columns matching that method
            if method_name is not None:
                # Handle both formats: "1_directinversion+p2p" and "sdxl_directinversion+p2p"
                # Extract the base method name (after first underscore or use full name)
                method_base = method_name.split('_', 1)[-1] if '_' in method_name else method_name
                method_part_base = method_part.split('_', 1)[-1] if '_' in method_part else method_part
                
                # Match if: exact match, or base names match, or method_part ends with method_base
                if (method_part == method_name or 
                    method_part_base == method_base or 
                    method_part.endswith(method_base) or
                    method_name.endswith(method_part_base)):
                    metrics_data[metric_name] = df[col]
            else:
                metrics_data[metric_name] = df[col]
        elif col == 'file_id':
            continue
        else:
            # Column without method prefix
            if method_name is None:
                metrics_data[col] = df[col]
    
    # Use method_name from CSV if not provided
    if method_name is None and len(df.columns) > 1:
        # Try to extract method name from first metric column
        first_col = [c for c in df.columns if c != 'file_id'][0]
        if '|' in first_col:
            method_name = first_col.split('|')[0]
    
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


def compare_two_methods(df1, df2, label1, label2, common_metrics):
    """
    Compare two dataframes and return results.
    """
    # Use the smaller dataframe as the base dataframe
    # Compare only the first n rows of the larger dataframe
    min_len = min(len(df1), len(df2))
    df1 = df1.head(min_len)
    df2 = df2.head(min_len)
    
    # Merge on file_id to ensure paired comparison
    merged = pd.merge(df1, df2, on='file_id', suffixes=('_1', '_2'))
    
    # Prepare results
    results = []
    summary_table = []
    
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
                winner = label1 if comparison['mean_diff'] > 0 else label2
            else:
                winner = label1 if comparison['mean_diff'] < 0 else label2
        else:
            winner = "No sig. diff."
        
        results.append({
            'metric': metric,
            'stats1': stats1,
            'stats2': stats2,
            'comparison': comparison,
            'winner': winner,
            'label1': label1,
            'label2': label2,
            'n_paired': len(merged)
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
    
    return results, summary_table, merged


def print_single_method_statistics(df, method_name, output=None):
    """Print statistics for a single method."""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    if output:
        sys.stdout = StringIO()
    
    metrics = [col for col in df.columns if col != 'file_id']
    
    print("=" * 80)
    print(f"STATISTICS FOR METHOD: {method_name}")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print()
    
    # Create statistics table
    stats_table = []
    for metric in metrics:
        stats = compute_statistics(df[metric])
        higher_is_better = METRIC_INFO.get(metric, (True, ""))[0]
        direction = "↑" if higher_is_better else "↓"
        
        stats_table.append([
            metric,
            f"{stats['mean']:.4f} ± {stats['std']:.4f}",
            f"{stats['median']:.4f}",
            f"[{stats['min']:.4f}, {stats['max']:.4f}]",
            stats['n'],
            direction
        ])
    
    headers = ["Metric", "Mean ± Std", "Median", "Range", "N", "Dir"]
    print(tabulate(stats_table, headers=headers, tablefmt="grid"))
    
    if output:
        output_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        with open(output, 'w') as f:
            f.write(output_str)
        print(f"\nResults saved to: {output}")
        print(output_str)
    else:
        sys.stdout = old_stdout


def compare_methods(csvs, methods=None, output=None, latex_output=None):
    """
    Main comparison function.
    
    Compare multiple CSV files with optional method names.
    
    Args:
        csvs: List of CSV file paths (required, at least 2 for comparison)
        methods: Optional list of method names (one per CSV file, or None to auto-detect)
        output: Optional output file path for text results
        latex_output: Optional output file path for LaTeX table code
    """
    if csvs is None or len(csvs) < 2:
        raise ValueError("Must provide at least 2 CSV files for comparison")
    
    # Use multiple files comparison mode
    return compare_multiple_methods(csvs, output, latex_output, method_names=methods)


def format_value_for_table(value, metric_name, scale_factor=1.0, decimals=2):
    """Format a value for the comparison table with appropriate scaling and precision."""
    scaled_value = value * scale_factor
    return f"{scaled_value:.{decimals}f}"


def calculate_percentage_change(baseline_value, current_value, higher_is_better):
    """Calculate percentage change relative to baseline."""
    if pd.isna(baseline_value) or pd.isna(current_value) or baseline_value == 0:
        return None, None
    
    if higher_is_better:
        # For metrics where higher is better, improvement means increase
        pct_change = ((current_value - baseline_value) / abs(baseline_value)) * 100
        direction = "↑" if pct_change > 0 else "↓"
    else:
        # For metrics where lower is better, improvement means decrease
        pct_change = ((baseline_value - current_value) / abs(baseline_value)) * 100
        direction = "↑" if pct_change > 0 else "↓"
    
    return pct_change, direction


def create_latex_table(dataframes, labels, common_metrics, baseline_idx=0):
    """
    Create a LaTeX table code for side-by-side comparison.
    
    Args:
        dataframes: List of dataframes
        labels: List of method labels (e.g., ['sd14', 'sd15', 'sd21'])
        common_metrics: List of common metrics to compare
        baseline_idx: Index of baseline method for percentage calculations
    """
    import os
    import re
    
    # Find common file_ids across all dataframes
    common_file_ids = set(dataframes[0]['file_id'])
    for df in dataframes[1:]:
        common_file_ids &= set(df['file_id'])
    common_file_ids = sorted(list(common_file_ids))
    
    # Filter all dataframes to common file_ids
    filtered_dfs = []
    for df in dataframes:
        filtered_df = df[df['file_id'].isin(common_file_ids)].copy()
        filtered_df = filtered_df.sort_values('file_id').reset_index(drop=True)
        filtered_dfs.append(filtered_df)
    
    # Calculate statistics for each method
    method_stats = []
    for df in filtered_dfs:
        stats = {}
        for metric in common_metrics:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors='coerce').dropna()
                if len(values) > 0:
                    stats[metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median()
                    }
                else:
                    stats[metric] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}
            else:
                stats[metric] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}
        method_stats.append(stats)
    
    # Define metric groups and formatting
    metric_groups = {
        'Structure': [
            ('structure_distance', 1000, 2, 'Distance$\\times10^3$ $\\downarrow$'),
        ],
        'Background Preservation': [
            ('psnr_unedit_part', 1, 2, 'PSNR $\\uparrow$'),
            ('lpips_unedit_part', 1000, 3, 'LPIPS$\\times10^3$ $\\downarrow$'),
            ('mse_unedit_part', 10000, 4, 'MSE$\\times10^4$ $\\downarrow$'),
            ('ssim_unedit_part', 100, 2, 'SSIM$\\times10^2$ $\\uparrow$'),
        ],
        'CLIP Similarity': [
            ('clip_similarity_target_image', 1, 2, 'Whole $\\uparrow$'),
            ('clip_similarity_target_image_edit_part', 1, 2, 'Edited $\\uparrow$'),
        ]
    }
    
    # Use labels directly as method names
    method_names = labels.copy()
    
    # Count total columns: Method column (Inverse/Editing) + metric columns
    num_metric_cols = sum(len(metrics_list) for metrics_list in metric_groups.values())
    total_cols = 1 + num_metric_cols  # Method + metrics
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append("% Required packages:")
    latex_lines.append("% \\usepackage{booktabs}  % for \\toprule, \\midrule, \\bottomrule, \\cmidrule")
    latex_lines.append("% \\usepackage{multirow}  % for \\multirow")
    latex_lines.append("% \\usepackage{xcolor}    % for \\textcolor")
    latex_lines.append("% \\usepackage{graphicx}  % for \\resizebox")
    latex_lines.append("")
    latex_lines.append("\\begin{table*}")
    latex_lines.append("\\centering")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append("\\begin{tabular}{" + "c" * total_cols + "}")
    latex_lines.append("\\toprule")
    
    # Header row 1: Method column + Main categories spanning metric columns
    header1_parts = ["\\textbf{Method}"]
    col_idx = 2
    for group_name, metrics_list in metric_groups.items():
        num_cols = len(metrics_list)
        header1_parts.append("\\multicolumn{" + str(num_cols) + "}{c}{\\textbf{" + group_name + "}}")
        col_idx += num_cols
    
    latex_lines.append(" & ".join(header1_parts) + " \\\\")
    latex_lines.append("\\cmidrule(lr){1-1} \\cmidrule(lr){2-" + str(total_cols) + "}")
    
    # Header row 2: Empty for Method column + metric names
    header2_parts = [""]
    for group_name, metrics_list in metric_groups.items():
        for metric_key, scale, decimals, display_name in metrics_list:
            header2_parts.append(display_name)
    
    latex_lines.append(" & ".join(header2_parts) + " \\\\")
    latex_lines.append("\\midrule")
    
    # First, determine the best method for each metric
    best_methods_per_metric = {}
    for group_name, metrics_list in metric_groups.items():
        for metric_key, scale, decimals, display_name in metrics_list:
            if metric_key not in common_metrics:
                continue
            
            higher_is_better = METRIC_INFO.get(metric_key, (True, ""))[0]
            best_idx = None
            best_val = None
            
            for idx in range(len(method_names)):
                if metric_key in method_stats[idx]:
                    mean_val = method_stats[idx][metric_key]['mean']
                    if not pd.isna(mean_val):
                        if best_val is None:
                            best_val = mean_val
                            best_idx = idx
                        else:
                            if higher_is_better:
                                if mean_val > best_val:
                                    best_val = mean_val
                                    best_idx = idx
                            else:
                                if mean_val < best_val:
                                    best_val = mean_val
                                    best_idx = idx
            
            best_methods_per_metric[metric_key] = best_idx
    
    # Data rows: one row per method
    for idx, method_name in enumerate(method_names):
        row_parts = [method_name]
        
        # Add values for each metric
        for group_name, metrics_list in metric_groups.items():
            for metric_key, scale, decimals, display_name in metrics_list:
                if metric_key not in common_metrics:
                    row_parts.append("")
                    continue
                
                if metric_key in method_stats[idx]:
                    mean_val = method_stats[idx][metric_key]['mean']
                    if not pd.isna(mean_val):
                        formatted_val = format_value_for_table(mean_val, metric_key, scale, decimals)
                        
                        # Check if this is the best method for this metric
                        is_best = (best_methods_per_metric.get(metric_key) == idx)
                        
                        # Calculate percentage change relative to baseline
                        if idx != baseline_idx:
                            baseline_mean = method_stats[baseline_idx][metric_key]['mean']
                            if not pd.isna(baseline_mean) and baseline_mean != 0:
                                higher_is_better = METRIC_INFO.get(metric_key, (True, ""))[0]
                                pct_change, direction = calculate_percentage_change(baseline_mean, mean_val, higher_is_better)
                                
                                if pct_change is not None:
                                    # Format: value with percentage change in red
                                    # Use 1 decimal place for small values, integer for larger ones
                                    abs_pct = abs(pct_change)
                                    if abs_pct < 1.0:
                                        pct_str = f"{abs_pct:.1f}"
                                    else:
                                        pct_str = f"{int(abs_pct)}"
                                    
                                    if direction == "↑":
                                        color = "\\textcolor{red}{" + pct_str + "\\%$\\uparrow$}"
                                    else:
                                        color = "\\textcolor{red}{" + pct_str + "\\%$\\downarrow$}"
                                    
                                    # Only bold if it's the best method
                                    if is_best:
                                        row_parts.append("\\textbf{" + formatted_val + "} (" + color + ")")
                                    else:
                                        row_parts.append(formatted_val + " (" + color + ")")
                                else:
                                    # Only bold if it's the best method
                                    if is_best:
                                        row_parts.append("\\textbf{" + formatted_val + "}")
                                    else:
                                        row_parts.append(formatted_val)
                            else:
                                # Only bold if it's the best method
                                if is_best:
                                    row_parts.append("\\textbf{" + formatted_val + "}")
                                else:
                                    row_parts.append(formatted_val)
                        else:
                            # Baseline method - only bold if it's the best
                            if is_best:
                                row_parts.append("\\textbf{" + formatted_val + "}")
                            else:
                                row_parts.append(formatted_val)
                    else:
                        row_parts.append("N/A")
                else:
                    row_parts.append("N/A")
        
        latex_lines.append(" & ".join(row_parts) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}%")
    latex_lines.append("}")
    latex_lines.append("\\caption{Comparison of different backbone models with direct inversion + p2p editing method.}")
    latex_lines.append("\\label{tab:backbone_comparison}")
    latex_lines.append("\\end{table*}")
    
    return "\n".join(latex_lines), len(common_file_ids)


def create_side_by_side_table(dataframes, labels, common_metrics, baseline_idx=0):
    """
    Create a side-by-side comparison table similar to the image format.
    Methods are shown as columns, metrics as rows.
    
    Args:
        dataframes: List of dataframes
        labels: List of method labels (e.g., ['sd14', 'sd15', 'sd21'])
        common_metrics: List of common metrics to compare
        baseline_idx: Index of baseline method for percentage calculations
    """
    import os
    import re
    
    # Find common file_ids across all dataframes
    common_file_ids = set(dataframes[0]['file_id'])
    for df in dataframes[1:]:
        common_file_ids &= set(df['file_id'])
    common_file_ids = sorted(list(common_file_ids))
    
    # Filter all dataframes to common file_ids
    filtered_dfs = []
    for df in dataframes:
        filtered_df = df[df['file_id'].isin(common_file_ids)].copy()
        filtered_df = filtered_df.sort_values('file_id').reset_index(drop=True)
        filtered_dfs.append(filtered_df)
    
    # Calculate statistics for each method
    method_stats = []
    for df in filtered_dfs:
        stats = {}
        for metric in common_metrics:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors='coerce').dropna()
                if len(values) > 0:
                    stats[metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median()
                    }
                else:
                    stats[metric] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}
            else:
                stats[metric] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}
        method_stats.append(stats)
    
    # Define metric groups and formatting
    metric_groups = {
        'Structure': [
            ('structure_distance', 1000, 2, 'Distance×10³ ↓'),  # Scale by 1000, 2 decimals
        ],
        'Background Preservation': [
            ('psnr_unedit_part', 1, 2, 'PSNR ↑'),
            ('lpips_unedit_part', 1000, 3, 'LPIPS×10³ ↓'),
            ('mse_unedit_part', 10000, 4, 'MSE×10⁴ ↓'),
            ('ssim_unedit_part', 100, 2, 'SSIM×10² ↑'),
        ],
        'CLIP Similarity': [
            ('clip_similarity_target_image', 1, 2, 'Whole ↑'),
            ('clip_similarity_target_image_edit_part', 1, 2, 'Edited ↑'),
        ]
    }
    
    # Use labels directly as method names
    method_names = labels.copy()
    
    # Build table structure: methods as rows, metrics as columns (to match LaTeX format)
    table_rows = []
    
    # Collect all metrics in order
    all_metrics_ordered = []
    for group_name, metrics_list in metric_groups.items():
        for metric_key, scale, decimals, display_name in metrics_list:
            if metric_key in common_metrics:
                all_metrics_ordered.append((metric_key, scale, decimals, display_name, group_name))
    
    # Header row 1: Method column + metric group names spanning their columns
    header1 = ['Method']
    current_col = 1
    group_spans = {}
    for group_name in metric_groups.keys():
        num_cols = len([m for m in all_metrics_ordered if m[4] == group_name])
        if num_cols > 0:
            group_spans[group_name] = (current_col, current_col + num_cols - 1)
            header1.extend([group_name] + [''] * (num_cols - 1))
            current_col += num_cols
    table_rows.append(header1)
    
    # Header row 2: Empty method column + metric display names
    header2 = ['']  # Empty for method column
    for metric_key, scale, decimals, display_name, group_name in all_metrics_ordered:
        header2.append(display_name)
    table_rows.append(header2)
    
    # Data rows: one row per method
    # Each row: [method name] + [value for metric1] + [value for metric2] + ...
    for idx, method_name in enumerate(method_names):
        # First column is the method name
        row = [method_name]
        
        # Add values for each metric (these go in the metric columns)
        for metric_key, scale, decimals, display_name, group_name in all_metrics_ordered:
            if metric_key in method_stats[idx]:
                mean_val = method_stats[idx][metric_key]['mean']
                if not pd.isna(mean_val):
                    formatted_val = format_value_for_table(mean_val, metric_key, scale, decimals)
                    
                    # Calculate percentage change relative to baseline
                    if idx != baseline_idx:
                        baseline_mean = method_stats[baseline_idx][metric_key]['mean']
                        if not pd.isna(baseline_mean) and baseline_mean != 0:
                            higher_is_better = METRIC_INFO.get(metric_key, (True, ""))[0]
                            pct_change, direction = calculate_percentage_change(baseline_mean, mean_val, higher_is_better)
                            
                            if pct_change is not None:
                                # Format: value with percentage change
                                # Use 1 decimal place for small values, integer for larger ones
                                abs_pct = abs(pct_change)
                                if abs_pct < 1.0:
                                    pct_str = f"{abs_pct:.1f}"
                                else:
                                    pct_str = f"{int(abs_pct)}"
                                row.append(f"{formatted_val} ({pct_str}%{direction})")
                            else:
                                row.append(formatted_val)
                        else:
                            row.append(formatted_val)
                    else:
                        # Baseline method - no percentage change
                        row.append(formatted_val)
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        
        table_rows.append(row)
    
    return table_rows, len(common_file_ids)


def compare_multiple_methods(csvs, output=None, latex_output=None, method_names=None):
    """
    Compare multiple CSV files pairwise and in side-by-side format.
    
    Args:
        csvs: List of CSV file paths
        output: Optional output file path for text results
        latex_output: Optional output file path for LaTeX table code
        method_names: Optional list of method names to extract from each CSV (one per CSV file)
    """
    import os
    
    # Load all dataframes
    dataframes = []
    labels = []
    
    if method_names is None:
        method_names = [None] * len(csvs)
    elif len(method_names) != len(csvs):
        raise ValueError(f"method_names length ({len(method_names)}) must match csvs length ({len(csvs)})")
    
    for i, csv_path in enumerate(csvs):
        method_name = method_names[i]
        df, name = load_and_extract(csv_path, method_name)
        
        # Use method_name if provided, otherwise use extracted name or filename
        if method_name is not None:
            label = method_name
        elif name is not None:
            # Try to extract method name (e.g., sd14, sd15, sd21)
            import re
            match = re.search(r'sd\d+', name.lower())
            if match:
                label = match.group().upper()
            else:
                label = name
        else:
            label = os.path.basename(csv_path).replace('.csv', '')
        
        dataframes.append(df)
        labels.append(label)
    
    # Find common metrics across all files
    all_metrics = [set(df.columns) - {'file_id'} for df in dataframes]
    common_metrics = sorted(set.intersection(*all_metrics))
    
    if not common_metrics:
        print("Warning: No common metrics found across all files!")
        return []
    
    print("=" * 80)
    print("STATISTICAL COMPARISON OF MULTIPLE IMAGE EDITING METHODS")
    print("=" * 80)
    print(f"\nComparing {len(csvs)} methods:")
    for i, (csv_path, label) in enumerate(zip(csvs, labels)):
        print(f"  {label}: {csv_path}")
    print(f"\nCommon metrics: {len(common_metrics)}")
    print()
    
    # Create side-by-side comparison table
    print("=" * 80)
    print("SIDE-BY-SIDE COMPARISON TABLE")
    print("=" * 80)
    print()
    
    table_rows, n_samples = create_side_by_side_table(dataframes, labels, common_metrics, baseline_idx=0)
    
    # Print the table
    # Note: The table structure is complex, so we'll print it manually
    # First, determine column widths
    if table_rows:
        num_cols = max(len(row) for row in table_rows)
        col_widths = [15] * num_cols  # Start with minimum width
        
        for row in table_rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)) + 2)  # Add padding
        
        # Print header rows
        if len(table_rows) >= 2:
            # Header row 1: Method column + metric group names
            header1 = table_rows[0]
            # Header row 2: Empty + metric display names
            header2 = table_rows[1]
            
            # Print header 1 with proper spacing
            header1_str = " | ".join(str(cell).center(col_widths[i]) if i < len(header1) else "".center(col_widths[i]) 
                                    for i, cell in enumerate(header1[:len(col_widths)]))
            print(header1_str)
            
            # Print separator
            separator = "-+-".join("-" * w for w in col_widths[:len(header1)])
            print(separator)
            
            # Print header 2
            header2_str = " | ".join(str(cell).center(col_widths[i]) if i < len(header2) else "".center(col_widths[i]) 
                                    for i, cell in enumerate(header2[:len(col_widths)]))
            print(header2_str)
            
            # Print separator
            print(separator)
            
            # Print data rows (methods as rows)
            for row in table_rows[2:]:
                row_str = " | ".join(str(cell).rjust(col_widths[i]) if i < len(row) else "".rjust(col_widths[i]) 
                                     for i, cell in enumerate(row[:len(col_widths)]))
                print(row_str)
    
    print(f"\nTotal samples: {n_samples}")
    print()
    
    # Generate LaTeX table if requested
    if latex_output:
        latex_code, n_samples_latex = create_latex_table(dataframes, labels, common_metrics, baseline_idx=0)
        with open(latex_output, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table code saved to: {latex_output}")
        print()
    
    # Also perform pairwise comparisons for detailed statistics
    print("=" * 80)
    print("PAIRWISE STATISTICAL COMPARISONS")
    print("=" * 80)
    print()
    
    all_results = []
    all_summaries = []
    
    for i in range(len(dataframes)):
        for j in range(i + 1, len(dataframes)):
            df1, df2 = dataframes[i], dataframes[j]
            label1, label2 = labels[i], labels[j]
            
            print("=" * 80)
            print(f"COMPARISON: {label1} vs {label2}")
            print("=" * 80)
            
            results, summary_table, merged = compare_two_methods(df1, df2, label1, label2, common_metrics)
            
            print(f"\nTotal paired samples: {len(merged)}")
            print()
            
            # Print summary table
            headers = ["Metric", label1, label2, "Diff", "p-value", "W1/W2/T", "Win%", "Winner"]
            print(tabulate(summary_table, headers=headers, tablefmt="grid"))
            print()
            
            # Store results
            all_results.append({
                'pair': (label1, label2),
                'results': results,
                'summary_table': summary_table,
                'n_paired': len(merged)
            })
            all_summaries.append((label1, label2, summary_table, len(merged)))
    
    # Print legend
    print("\n" + "=" * 80)
    print("LEGEND")
    print("=" * 80)
    print("↑ = Higher is better, ↓ = Lower is better")
    print("Diff = Mean(Method1) - Mean(Method2)")
    print("W1/W2/T = Wins for Method1 / Wins for Method2 / Ties")
    print("Win% = Percentage of images where Method 1 is better")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print("Percentage changes in side-by-side table are relative to the first method (baseline)")
    
    # Overall summary across all pairs
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY ACROSS ALL PAIRS")
    print("=" * 80)
    
    # Count wins for each method across all comparisons
    method_wins = {label: 0 for label in labels}
    method_losses = {label: 0 for label in labels}
    method_ties = {label: 0 for label in labels}
    
    for pair_result in all_results:
        label1, label2 = pair_result['pair']
        for r in pair_result['results']:
            winner = r['winner']
            if winner == label1:
                method_wins[label1] += 1
                method_losses[label2] += 1
            elif winner == label2:
                method_wins[label2] += 1
                method_losses[label1] += 1
            else:
                method_ties[label1] += 1
                method_ties[label2] += 1
    
    print("\nTotal metric wins per method (across all pairwise comparisons):")
    for label in labels:
        print(f"  {label}: {method_wins[label]} wins, {method_losses[label]} losses, {method_ties[label]} ties")
    
    # Save to file if requested
    if output:
        import sys
        from io import StringIO
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Re-run print statements
        print("=" * 80)
        print("STATISTICAL COMPARISON OF MULTIPLE IMAGE EDITING METHODS")
        print("=" * 80)
        print(f"\nComparing {len(csvs)} methods:")
        for i, (csv_path, label) in enumerate(zip(csvs, labels)):
            print(f"  {label}: {csv_path}")
        print(f"\nCommon metrics: {len(common_metrics)}")
        print()
        
        # Side-by-side table
        print("=" * 80)
        print("SIDE-BY-SIDE COMPARISON TABLE")
        print("=" * 80)
        print()
        
        table_rows, n_samples = create_side_by_side_table(dataframes, labels, common_metrics, baseline_idx=0)
        if table_rows:
            num_cols = max(len(row) for row in table_rows)
            col_widths = [15] * num_cols  # Start with minimum width
            
            for row in table_rows:
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        col_widths[i] = max(col_widths[i], len(str(cell)) + 2)  # Add padding
            
            if len(table_rows) >= 2:
                header1 = table_rows[0]
                header2 = table_rows[1]
                
                header1_str = " | ".join(str(cell).center(col_widths[i]) if i < len(header1) else "".center(col_widths[i]) 
                                        for i, cell in enumerate(header1[:len(col_widths)]))
                print(header1_str)
                
                separator = "-+-".join("-" * w for w in col_widths[:len(header1)])
                print(separator)
                
                header2_str = " | ".join(str(cell).center(col_widths[i]) if i < len(header2) else "".center(col_widths[i]) 
                                        for i, cell in enumerate(header2[:len(col_widths)]))
                print(header2_str)
                
                print(separator)
                
                for row in table_rows[2:]:
                    row_str = " | ".join(str(cell).rjust(col_widths[i]) if i < len(row) else "".rjust(col_widths[i]) 
                                         for i, cell in enumerate(row[:len(col_widths)]))
                    print(row_str)
        
        print(f"\nTotal samples: {n_samples}")
        print()
        
        # Generate LaTeX table if requested
        if latex_output:
            latex_code, n_samples_latex = create_latex_table(dataframes, labels, common_metrics, baseline_idx=0)
            with open(latex_output, 'w') as f:
                f.write(latex_code)
            print(f"LaTeX table code saved to: {latex_output}")
            print()
        
        for label1, label2, summary_table, n_paired in all_summaries:
            print("=" * 80)
            print(f"COMPARISON: {label1} vs {label2}")
            print("=" * 80)
            print(f"\nTotal paired samples: {n_paired}")
            print()
            headers = ["Metric", label1, label2, "Diff", "p-value", "W1/W2/T", "Win%", "Winner"]
            print(tabulate(summary_table, headers=headers, tablefmt="grid"))
            print()
        
        print("=" * 80)
        print("OVERALL SUMMARY ACROSS ALL PAIRS")
        print("=" * 80)
        print("\nTotal metric wins per method (across all pairwise comparisons):")
        for label in labels:
            print(f"  {label}: {method_wins[label]} wins, {method_losses[label]} losses, {method_ties[label]} ties")
        
        output_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(output, 'w') as f:
            f.write(output_str)
        print(f"\nResults saved to: {output}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare image editing methods statistically")
    parser.add_argument('--csvs', type=str, nargs='+', required=True,
                        help="Paths to CSV files for comparison (at least 2 required)")
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help="Optional method names (one per CSV file, auto-detected if not provided)")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save comparison results")
    parser.add_argument('--latex', type=str, default=None,
                        help="Path to save LaTeX table code")
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.csvs) < 2:
        parser.error("Must provide at least 2 CSV files for comparison")
    
    if args.methods is not None and len(args.methods) != len(args.csvs):
        parser.error(f"Number of methods ({len(args.methods)}) must match number of CSV files ({len(args.csvs)})")
    
    # Run comparison
    compare_methods(csvs=args.csvs, methods=args.methods, output=args.output, latex_output=args.latex)


if __name__ == "__main__":
    main()
