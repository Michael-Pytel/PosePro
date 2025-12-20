import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def calculate_velocity(angle_series):
    """Calculate velocity (first derivative) of angle series"""
    velocity = angle_series.diff()
    return velocity

def calculate_acceleration(velocity_series):
    """Calculate acceleration (second derivative) of angle series"""
    acceleration = velocity_series.diff()
    return acceleration

def calculate_jerk(acceleration_series):
    """Calculate jerk (third derivative) of angle series"""
    jerk = acceleration_series.diff()
    return jerk

def select_best_side(df):
    """
    Determine which side (left or right) is more visible and create unified columns.
    Returns a new dataframe with 'elbow_angle', 'hip_angle', 'spine_angle' columns
    using the better side.
    """
    df_unified = df.copy()
    
    # Define pairs of angles to unify
    angle_pairs = [
        ('left_elbow_angle', 'right_elbow_angle', 'elbow_angle'),
        ('left_hip_angle', 'right_hip_angle', 'hip_angle'),
        ('left_spine_angle', 'right_spine_angle', 'spine_angle')
    ]
    
    for left_col, right_col, unified_col in angle_pairs:
        if left_col not in df.columns or right_col not in df.columns:
            continue
        
        # Calculate validity percentage for each side
        left_valid_pct = df[left_col].notna().sum() / len(df)
        right_valid_pct = df[right_col].notna().sum() / len(df)
        
        # Choose the side with more valid data
        if left_valid_pct >= right_valid_pct:
            df_unified[unified_col] = df[left_col]
        else:
            df_unified[unified_col] = df[right_col]
    
    # Keep head_tilt_angle as is (not paired)
    if 'head_tilt_angle' in df.columns:
        df_unified['head_tilt_angle'] = df['head_tilt_angle']
    
    return df_unified

def get_lowest_highest_frames(df):
    """
    Find the frames where the pushup is at lowest and highest points
    Based on the unified elbow angle
    
    Returns:
        lowest_frame_idx, highest_frame_idx
    """
    if 'elbow_angle' not in df.columns:
        return None, None
    
    elbow_angle = df['elbow_angle']
    
    if elbow_angle.notna().sum() == 0:
        return None, None
    
    # Lowest point = minimum angle (most bent)
    lowest_idx = elbow_angle.idxmin()
    # Highest point = maximum angle (most extended)
    highest_idx = elbow_angle.idxmax()
    
    return lowest_idx, highest_idx

def calculate_basic_stats(angle_series, valid_threshold=0.5):
    """
    Calculate basic statistics for an angle series
    
    Returns dict with stats or NaN values if insufficient data
    """
    valid_pct = angle_series.notna().sum() / len(angle_series)
    
    stats = {
        'valid_pct': valid_pct * 100
    }
    
    # If less than threshold% valid data, return NaN for all stats
    if valid_pct < valid_threshold:
        stats.update({
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'range': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'cv': np.nan
        })
        return stats
    
    # Calculate stats from valid values only
    valid_values = angle_series.dropna()
    
    mean_val = valid_values.mean()
    std_val = valid_values.std()
    
    stats.update({
        'mean': mean_val,
        'std': std_val,
        'min': valid_values.min(),
        'max': valid_values.max(),
        'range': valid_values.max() - valid_values.min(),
        'median': valid_values.median(),
        'q25': valid_values.quantile(0.25),
        'q75': valid_values.quantile(0.75),
        'cv': std_val / mean_val if mean_val != 0 else np.nan
    })
    
    return stats

def calculate_dynamics(angle_series, valid_threshold=0.5):
    """
    Calculate velocity, acceleration, and smoothness (jerk) metrics
    
    Returns dict with dynamics stats
    """
    valid_pct = angle_series.notna().sum() / len(angle_series)
    
    if valid_pct < valid_threshold:
        return {
            'mean_velocity': np.nan,
            'max_velocity': np.nan,
            'mean_acceleration': np.nan,
            'smoothness_jerk': np.nan
        }
    
    # Calculate derivatives
    velocity = calculate_velocity(angle_series)
    acceleration = calculate_acceleration(velocity)
    jerk = calculate_jerk(acceleration)
    
    # Calculate statistics from valid values
    velocity_valid = velocity.dropna()
    acceleration_valid = acceleration.dropna()
    jerk_valid = jerk.dropna()
    
    return {
        'mean_velocity': velocity_valid.abs().mean() if len(velocity_valid) > 0 else np.nan,
        'max_velocity': velocity_valid.abs().max() if len(velocity_valid) > 0 else np.nan,
        'mean_acceleration': acceleration_valid.abs().mean() if len(acceleration_valid) > 0 else np.nan,
        'smoothness_jerk': jerk_valid.abs().mean() if len(jerk_valid) > 0 else np.nan
    }

def calculate_critical_moments(angle_series, lowest_idx, highest_idx, valid_threshold=0.5):
    """
    Calculate angle values at critical moments (lowest and highest points)
    """
    valid_pct = angle_series.notna().sum() / len(angle_series)
    
    if valid_pct < valid_threshold or lowest_idx is None or highest_idx is None:
        return {
            'at_lowest': np.nan,
            'at_highest': np.nan
        }
    
    return {
        'at_lowest': angle_series.loc[lowest_idx] if pd.notna(angle_series.loc[lowest_idx]) else np.nan,
        'at_highest': angle_series.loc[highest_idx] if pd.notna(angle_series.loc[highest_idx]) else np.nan
    }

def calculate_pushup_specific_features(df, valid_threshold=0.5):
    """
    Calculate pushup-specific features like time in bottom position, descent/ascent rates
    """
    if 'elbow_angle' not in df.columns:
        return {
            'bottom_position_time_pct': np.nan,
            'bottom_position_time_frames': np.nan,
            'descent_rate': np.nan,
            'ascent_rate': np.nan,
            'descent_duration': np.nan,
            'ascent_duration': np.nan,
            'movement_asymmetry': np.nan
        }
    
    elbow_angle = df['elbow_angle']
    
    valid_pct = elbow_angle.notna().sum() / len(elbow_angle)
    
    if valid_pct < valid_threshold:
        return {
            'bottom_position_time_pct': np.nan,
            'bottom_position_time_frames': np.nan,
            'descent_rate': np.nan,
            'ascent_rate': np.nan,
            'descent_duration': np.nan,
            'ascent_duration': np.nan,
            'movement_asymmetry': np.nan
        }
    
    # Find lowest and highest points
    if elbow_angle.notna().sum() == 0:
        return {
            'bottom_position_time_pct': np.nan,
            'bottom_position_time_frames': np.nan,
            'descent_rate': np.nan,
            'ascent_rate': np.nan,
            'descent_duration': np.nan,
            'ascent_duration': np.nan,
            'movement_asymmetry': np.nan
        }
    
    lowest_idx = elbow_angle.idxmin()
    highest_idx = elbow_angle.idxmax()
    
    # Time in bottom position (elbow angle < 90 degrees)
    bottom_mask = elbow_angle < 90
    bottom_frames = bottom_mask.sum()
    bottom_pct = (bottom_frames / len(elbow_angle)) * 100
    
    # Descent and ascent phases
    # Split at lowest point
    descent_phase = elbow_angle.loc[:lowest_idx]
    ascent_phase = elbow_angle.loc[lowest_idx:]
    
    # Calculate rates (change in angle per frame)
    descent_values = descent_phase.dropna()
    ascent_values = ascent_phase.dropna()
    
    if len(descent_values) > 1:
        descent_rate = abs(descent_values.iloc[-1] - descent_values.iloc[0]) / len(descent_values)
        descent_duration = len(descent_values)
    else:
        descent_rate = np.nan
        descent_duration = np.nan
    
    if len(ascent_values) > 1:
        ascent_rate = abs(ascent_values.iloc[-1] - ascent_values.iloc[0]) / len(ascent_values)
        ascent_duration = len(ascent_values)
    else:
        ascent_rate = np.nan
        ascent_duration = np.nan
    
    # Movement asymmetry (difference between descent and ascent rates)
    if pd.notna(descent_rate) and pd.notna(ascent_rate):
        movement_asymmetry = abs(descent_rate - ascent_rate)
    else:
        movement_asymmetry = np.nan
    
    return {
        'bottom_position_time_pct': bottom_pct,
        'bottom_position_time_frames': bottom_frames,
        'descent_rate': descent_rate,
        'ascent_rate': ascent_rate,
        'descent_duration': descent_duration,
        'ascent_duration': ascent_duration,
        'movement_asymmetry': movement_asymmetry
    }

def aggregate_single_file(csv_path, valid_threshold=0.5):
    """
    Aggregate all features from a single CSV file
    
    Returns dict with all aggregated features
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Get file ID (filename without .csv)
        file_id = csv_path.stem
        
        # Select best side for paired angles
        df = select_best_side(df)
        
        # Initialize result dict
        result = {'id': file_id}
        
        # Unified angle columns (after selecting best side)
        angle_columns = [
            'elbow_angle',
            'hip_angle',
            'spine_angle',
            'head_tilt_angle'
        ]
        
        # Find lowest and highest frames
        lowest_idx, highest_idx = get_lowest_highest_frames(df)
        
        # Process each angle
        for angle_col in angle_columns:
            if angle_col not in df.columns:
                continue
            
            angle_series = df[angle_col]
            
            # Basic statistics
            basic_stats = calculate_basic_stats(angle_series, valid_threshold)
            for stat_name, stat_value in basic_stats.items():
                result[f'{angle_col}_{stat_name}'] = stat_value
            
            # Dynamics (velocity, acceleration, jerk)
            dynamics = calculate_dynamics(angle_series, valid_threshold)
            for dyn_name, dyn_value in dynamics.items():
                result[f'{angle_col}_{dyn_name}'] = dyn_value
            
            # Critical moments
            critical = calculate_critical_moments(angle_series, lowest_idx, highest_idx, valid_threshold)
            for crit_name, crit_value in critical.items():
                result[f'{angle_col}_{crit_name}'] = crit_value
        
        # Pushup-specific features (using unified elbow_angle)
        pushup_features = calculate_pushup_specific_features(df, valid_threshold)
        result.update(pushup_features)
        
        return result
        
    except Exception as e:
        print(f"Error processing {csv_path.name}: {str(e)}")
        return None

def aggregate_folder(input_folder, output_csv, valid_threshold=0.5, verbose=True):
    """
    Aggregate all CSV files in a folder into a single CSV
    
    Args:
        input_folder: Path to folder containing CSV files with angles
        output_csv: Path to output CSV file
        valid_threshold: Minimum fraction of valid (non-NaN) values required
        verbose: Whether to print progress
    """
    input_path = Path(input_folder)
    
    # Find all CSV files
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {input_folder}")
        print(f"Valid threshold: {valid_threshold*100}%")
        print("Using best visible side for paired angles (elbow, hip, spine)")
        print("="*60)
    
    # Process each file
    all_results = []
    
    for csv_file in tqdm(csv_files, desc="Aggregating files", disable=not verbose):
        result = aggregate_single_file(csv_file, valid_threshold)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No files were successfully processed")
        return
    
    # Create DataFrame
    df_aggregated = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_aggregated.to_csv(output_path, index=False)
    
    if verbose:
        print("\n" + "="*60)
        print("AGGREGATION COMPLETE")
        print("="*60)
        print(f"Successfully aggregated: {len(all_results)}/{len(csv_files)} files")
        print(f"Output shape: {df_aggregated.shape}")
        print(f"Output saved to: {output_path}")
        print("\nFeature summary:")
        print(f"  Total columns: {len(df_aggregated.columns)}")
        print(f"  Features per sample: {len(df_aggregated.columns) - 1}")  # -1 for ID
        
        # Check for missing data
        nan_counts = df_aggregated.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nColumns with NaN values:")
            nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
            for col, count in nan_cols.head(10).items():
                pct = (count / len(df_aggregated)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate angle features from multiple CSV files into a single dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input folder containing CSV files with angle data'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output CSV file path for aggregated features'
    )
    
    parser.add_argument(
        '--valid-threshold',
        type=float,
        default=0.5,
        help='Minimum fraction of valid (non-NaN) values required (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.valid_threshold <= 1.0:
        parser.error("--valid-threshold must be between 0.0 and 1.0")
    
    # Aggregate
    aggregate_folder(
        input_folder=args.input,
        output_csv=args.output,
        valid_threshold=args.valid_threshold,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()