import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def get_landmark_columns(landmark_idx, include_z=False):
    """
    Get column indices for a landmark in flat format
    
    Args:
        landmark_idx: landmark index (0-32)
        include_z: if True, return (x_col, y_col, z_col), else (x_col, y_col)
    
    Returns:
        tuple of column indices (accounting for frame and timestamp)
    """
    base_idx = landmark_idx * 4 + 2  # +2 to skip 'frame' and 'timestamp'
    
    x_col = base_idx
    y_col = base_idx + 1
    z_col = base_idx + 2
    vis_col = base_idx + 3
    
    if include_z:
        return x_col, y_col, z_col
    else:
        return x_col, y_col

def extract_landmark_vectors(df, landmark_idx, include_z=False):
    """
    Extract x, y (and optionally z) vectors for a landmark across all frames
    
    Args:
        df: pandas DataFrame
        landmark_idx: landmark index (0-32)
        include_z: if True, return x, y, z arrays
    
    Returns:
        numpy arrays for x, y (and z if requested)
    """
    cols = get_landmark_columns(landmark_idx, include_z)
    
    if include_z:
        x_col, y_col, z_col = cols
        return df.iloc[:, x_col].values, df.iloc[:, y_col].values, df.iloc[:, z_col].values
    else:
        x_col, y_col = cols
        return df.iloc[:, x_col].values, df.iloc[:, y_col].values
    
def extract_landmark_with_visibility(df, landmark_idx, include_z=False):
    """
    Extract x, y, z (optional) and visibility vectors for a landmark
    
    Returns:
        if include_z: (x, y, z, visibility)
        else: (x, y, visibility)
    """
    base_idx = landmark_idx * 4 + 2  # +2 to skip 'frame' and 'timestamp'
    
    x = df.iloc[:, base_idx].values
    y = df.iloc[:, base_idx + 1].values
    z = df.iloc[:, base_idx + 2].values if include_z else None
    vis = df.iloc[:, base_idx + 3].values
    
    if include_z:
        return x, y, z, vis
    else:
        return x, y, vis
    
def calculate_angles_vectorized_2d(ax, ay, bx, by, cx, cy):
    """
    Calculate angles at point b for all frames simultaneously (2D)
    
    Args:
        ax, ay: x, y coordinates of point a (numpy arrays)
        bx, by: x, y coordinates of point b (numpy arrays)
        cx, cy: x, y coordinates of point c (numpy arrays)
    
    Returns:
        numpy array of angles in degrees
    """
    # Vectors ba and bc
    ba_x = ax - bx
    ba_y = ay - by
    
    bc_x = cx - bx
    bc_y = cy - by
    
    # Dot product
    dot_product = ba_x * bc_x + ba_y * bc_y
    
    # Magnitudes
    mag_ba = np.sqrt(ba_x**2 + ba_y**2)
    mag_bc = np.sqrt(bc_x**2 + bc_y**2)
    
    # Avoid division by zero
    mag_product = mag_ba * mag_bc
    mag_product = np.where(mag_product == 0, 1e-10, mag_product)
    
    # Cosine of angle
    cos_angle = np.clip(dot_product / mag_product, -1.0, 1.0)
    
    # Angle in degrees
    angles = np.degrees(np.arccos(cos_angle))
    
    return angles

def calculate_angles_vectorized_2d_with_visibility(ax, ay, avis, bx, by, bvis, cx, cy, cvis, min_visibility=0.5):
    """
    Calculate angles at point b for all frames with visibility filtering
    
    Args:
        ax, ay, avis: x, y, visibility of point a
        bx, by, bvis: x, y, visibility of point b
        cx, cy, cvis: x, y, visibility of point c
        min_visibility: minimum visibility threshold (default 0.5 = 50%)
    
    Returns:
        numpy array of angles in degrees (with np.nan for low visibility)
    """
    # Vectors ba and bc
    ba_x = ax - bx
    ba_y = ay - by
    
    bc_x = cx - bx
    bc_y = cy - by
    
    # Dot product
    dot_product = ba_x * bc_x + ba_y * bc_y
    
    # Magnitudes
    mag_ba = np.sqrt(ba_x**2 + ba_y**2)
    mag_bc = np.sqrt(bc_x**2 + bc_y**2)
    
    # Avoid division by zero
    mag_product = mag_ba * mag_bc
    mag_product = np.where(mag_product == 0, 1e-10, mag_product)
    
    # Cosine of angle
    cos_angle = np.clip(dot_product / mag_product, -1.0, 1.0)
    
    # Angle in degrees
    angles = np.degrees(np.arccos(cos_angle))
    
    # Check minimum visibility of all three points
    min_vis = np.minimum(np.minimum(avis, bvis), cvis)
    
    # Set angles to NaN where visibility is below threshold
    angles = np.where(min_vis >= min_visibility, angles, np.nan)
    
    return angles

def calculate_angles_vectorized_3d_with_visibility(ax, ay, az, avis, bx, by, bz, bvis, cx, cy, cz, cvis, min_visibility=0.5):
    """
    Calculate angles at point b for all frames with visibility filtering (3D)
    """
    # Vectors ba and bc
    ba_x, ba_y, ba_z = ax - bx, ay - by, az - bz
    bc_x, bc_y, bc_z = cx - bx, cy - by, cz - bz
    
    # Dot product
    dot_product = ba_x * bc_x + ba_y * bc_y + ba_z * bc_z
    
    # Magnitudes
    mag_ba = np.sqrt(ba_x**2 + ba_y**2 + ba_z**2)
    mag_bc = np.sqrt(bc_x**2 + bc_y**2 + bc_z**2)
    
    # Avoid division by zero
    mag_product = mag_ba * mag_bc
    mag_product = np.where(mag_product == 0, 1e-10, mag_product)
    
    # Cosine of angle
    cos_angle = np.clip(dot_product / mag_product, -1.0, 1.0)
    
    # Angle in degrees
    angles = np.degrees(np.arccos(cos_angle))
    
    # Check minimum visibility of all three points
    min_vis = np.minimum(np.minimum(avis, bvis), cvis)
    
    # Set angles to NaN where visibility is below threshold
    angles = np.where(min_vis >= min_visibility, angles, np.nan)
    
    return angles

def calculate_angles_vectorized_3d(ax, ay, az, bx, by, bz, cx, cy, cz):
    """
    Calculate angles at point b for all frames simultaneously (3D)
    """
    # Vectors ba and bc
    ba_x, ba_y, ba_z = ax - bx, ay - by, az - bz
    bc_x, bc_y, bc_z = cx - bx, cy - by, cz - bz
    
    # Dot product
    dot_product = ba_x * bc_x + ba_y * bc_y + ba_z * bc_z
    
    # Magnitudes
    mag_ba = np.sqrt(ba_x**2 + ba_y**2 + ba_z**2)
    mag_bc = np.sqrt(bc_x**2 + bc_y**2 + bc_z**2)
    
    # Avoid division by zero
    mag_product = mag_ba * mag_bc
    mag_product = np.where(mag_product == 0, 1e-10, mag_product)
    
    # Cosine of angle
    cos_angle = np.clip(dot_product / mag_product, -1.0, 1.0)
    
    # Angle in degrees
    angles = np.degrees(np.arccos(cos_angle))
    
    return angles

def calculate_head_tilt_vectorized(df):
    """
    Calculate head tilt angle for all frames
    """
    # Extract ear positions (average of left and right)
    left_ear_x, left_ear_y = extract_landmark_vectors(df, 7)
    right_ear_x, right_ear_y = extract_landmark_vectors(df, 8)
    ear_x = (left_ear_x + right_ear_x) / 2
    ear_y = (left_ear_y + right_ear_y) / 2
    
    # Extract nose
    nose_x, nose_y = extract_landmark_vectors(df, 0)
    
    # Extract shoulders (average)
    left_shoulder_x, left_shoulder_y = extract_landmark_vectors(df, 11)
    right_shoulder_x, right_shoulder_y = extract_landmark_vectors(df, 12)
    shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    
    # Head vector: ear → nose
    head_x = nose_x - ear_x
    head_y = nose_y - ear_y
    
    # Neck vector: ear → shoulder
    neck_x = shoulder_x - ear_x
    neck_y = shoulder_y - ear_y
    
    # Dot product
    dot_product = head_x * neck_x + head_y * neck_y
    
    # Magnitudes
    mag_head = np.sqrt(head_x**2 + head_y**2)
    mag_neck = np.sqrt(neck_x**2 + neck_y**2)
    
    # Avoid division by zero
    mag_product = mag_head * mag_neck
    mag_product = np.where(mag_product == 0, 1e-10, mag_product)
    
    # Cosine and angle
    cos_angle = np.clip(dot_product / mag_product, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))
    
    # Determine sign (cross product for 2D)
    cross = head_x * neck_y - head_y * neck_x
    angles = np.where(cross < 0, -angles, angles)
    
    return angles

def calculate_head_tilt_vectorized_with_visibility(df, min_visibility=0.5):
    """
    Calculate head tilt angle for all frames with visibility filtering.
    Uses the most visible landmark when pairs are occluded (e.g., side view).
    Averages only when BOTH landmarks exceed the visibility threshold.
    """
    # Extract ear positions with visibility
    left_ear_x, left_ear_y, left_ear_vis = extract_landmark_with_visibility(df, 7)
    right_ear_x, right_ear_y, right_ear_vis = extract_landmark_with_visibility(df, 8)
    
    # Use visibility-based selection for ears
    both_ears_above_threshold = (left_ear_vis >= min_visibility) & (right_ear_vis >= min_visibility)
    use_left_ear = left_ear_vis >= right_ear_vis
    
    ear_x = np.where(both_ears_above_threshold,
                     (left_ear_x + right_ear_x) / 2,  # Average only when BOTH above threshold
                     np.where(use_left_ear, left_ear_x, right_ear_x))  # Use better one otherwise
    ear_y = np.where(both_ears_above_threshold,
                     (left_ear_y + right_ear_y) / 2,
                     np.where(use_left_ear, left_ear_y, right_ear_y))
    ear_vis = np.maximum(left_ear_vis, right_ear_vis)  # Take BEST visibility
    
    # Extract nose
    nose_x, nose_y, nose_vis = extract_landmark_with_visibility(df, 0)
    
    # Extract shoulders with visibility-based selection
    left_shoulder_x, left_shoulder_y, left_shoulder_vis = extract_landmark_with_visibility(df, 11)
    right_shoulder_x, right_shoulder_y, right_shoulder_vis = extract_landmark_with_visibility(df, 12)
    
    both_shoulders_above_threshold = (left_shoulder_vis >= min_visibility) & (right_shoulder_vis >= min_visibility)
    use_left_shoulder = left_shoulder_vis >= right_shoulder_vis
    
    shoulder_x = np.where(both_shoulders_above_threshold,
                         (left_shoulder_x + right_shoulder_x) / 2,  # Average only when BOTH above threshold
                         np.where(use_left_shoulder, left_shoulder_x, right_shoulder_x))
    shoulder_y = np.where(both_shoulders_above_threshold,
                         (left_shoulder_y + right_shoulder_y) / 2,
                         np.where(use_left_shoulder, left_shoulder_y, right_shoulder_y))
    shoulder_vis = np.maximum(left_shoulder_vis, right_shoulder_vis)
    
    # Head vector: ear → nose
    head_x = nose_x - ear_x
    head_y = nose_y - ear_y
    
    # Neck vector: ear → shoulder
    neck_x = shoulder_x - ear_x
    neck_y = shoulder_y - ear_y
    
    # Dot product
    dot_product = head_x * neck_x + head_y * neck_y
    
    # Magnitudes
    mag_head = np.sqrt(head_x**2 + head_y**2)
    mag_neck = np.sqrt(neck_x**2 + neck_y**2)
    
    # Avoid division by zero
    mag_product = mag_head * mag_neck
    mag_product = np.where(mag_product == 0, 1e-10, mag_product)
    
    # Cosine and angle
    cos_angle = np.clip(dot_product / mag_product, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))
    
    # Determine sign (cross product for 2D)
    cross = head_x * neck_y - head_y * neck_x
    angles = np.where(cross < 0, -angles, angles)
    
    # Check minimum visibility (using the BEST visibility from each pair)
    min_vis = np.minimum(np.minimum(ear_vis, nose_vis), shoulder_vis)
    angles = np.where(min_vis >= min_visibility, angles, np.nan)
    
    return angles
    
def add_angle_features(df, use_3d=False, min_visibility=0.5):
    """
    Calculate all angle features with visibility filtering and add them as new columns
    
    Args:
        df: pandas DataFrame with pose landmarks
        use_3d: if True, use x, y, z; if False, use only x, y
        min_visibility: minimum visibility threshold (0.0 to 1.0)
    
    Returns:
        DataFrame with added angle columns (angles are NaN where visibility < threshold)
    """
    df = df.copy()
    
    if use_3d:
        # Left elbow angle: shoulder(11) - elbow(13) - wrist(15)
        s11_x, s11_y, s11_z, s11_vis = extract_landmark_with_visibility(df, 11, include_z=True)
        e13_x, e13_y, e13_z, e13_vis = extract_landmark_with_visibility(df, 13, include_z=True)
        w15_x, w15_y, w15_z, w15_vis = extract_landmark_with_visibility(df, 15, include_z=True)
        df['left_elbow_angle'] = calculate_angles_vectorized_3d_with_visibility(
            s11_x, s11_y, s11_z, s11_vis,
            e13_x, e13_y, e13_z, e13_vis,
            w15_x, w15_y, w15_z, w15_vis,
            min_visibility
        )
        
        # Right elbow angle: shoulder(12) - elbow(14) - wrist(16)
        s12_x, s12_y, s12_z, s12_vis = extract_landmark_with_visibility(df, 12, include_z=True)
        e14_x, e14_y, e14_z, e14_vis = extract_landmark_with_visibility(df, 14, include_z=True)
        w16_x, w16_y, w16_z, w16_vis = extract_landmark_with_visibility(df, 16, include_z=True)
        df['right_elbow_angle'] = calculate_angles_vectorized_3d_with_visibility(
            s12_x, s12_y, s12_z, s12_vis,
            e14_x, e14_y, e14_z, e14_vis,
            w16_x, w16_y, w16_z, w16_vis,
            min_visibility
        )
        
        # Left hip angle: shoulder(11) - hip(23) - knee(25)
        h23_x, h23_y, h23_z, h23_vis = extract_landmark_with_visibility(df, 23, include_z=True)
        k25_x, k25_y, k25_z, k25_vis = extract_landmark_with_visibility(df, 25, include_z=True)
        df['left_hip_angle'] = calculate_angles_vectorized_3d_with_visibility(
            s11_x, s11_y, s11_z, s11_vis,
            h23_x, h23_y, h23_z, h23_vis,
            k25_x, k25_y, k25_z, k25_vis,
            min_visibility
        )
        
        # Right hip angle: shoulder(12) - hip(24) - knee(26)
        h24_x, h24_y, h24_z, h24_vis = extract_landmark_with_visibility(df, 24, include_z=True)
        k26_x, k26_y, k26_z, k26_vis = extract_landmark_with_visibility(df, 26, include_z=True)
        df['right_hip_angle'] = calculate_angles_vectorized_3d_with_visibility(
            s12_x, s12_y, s12_z, s12_vis,
            h24_x, h24_y, h24_z, h24_vis,
            k26_x, k26_y, k26_z, k26_vis,
            min_visibility
        )
        
        # Left spine alignment: shoulder(11) - hip(23) - ankle(27)
        a27_x, a27_y, a27_z, a27_vis = extract_landmark_with_visibility(df, 27, include_z=True)
        df['left_spine_angle'] = calculate_angles_vectorized_3d_with_visibility(
            s11_x, s11_y, s11_z, s11_vis,
            h23_x, h23_y, h23_z, h23_vis,
            a27_x, a27_y, a27_z, a27_vis,
            min_visibility
        )
        
        # Right spine alignment: shoulder(12) - hip(24) - ankle(28)
        a28_x, a28_y, a28_z, a28_vis = extract_landmark_with_visibility(df, 28, include_z=True)
        df['right_spine_angle'] = calculate_angles_vectorized_3d_with_visibility(
            s12_x, s12_y, s12_z, s12_vis,
            h24_x, h24_y, h24_z, h24_vis,
            a28_x, a28_y, a28_z, a28_vis,
            min_visibility
        )
        
    else:
        # 2D version (only x, y)
        # Left elbow angle
        s11_x, s11_y, s11_vis = extract_landmark_with_visibility(df, 11)
        e13_x, e13_y, e13_vis = extract_landmark_with_visibility(df, 13)
        w15_x, w15_y, w15_vis = extract_landmark_with_visibility(df, 15)
        df['left_elbow_angle'] = calculate_angles_vectorized_2d_with_visibility(
            s11_x, s11_y, s11_vis,
            e13_x, e13_y, e13_vis,
            w15_x, w15_y, w15_vis,
            min_visibility
        )
        
        # Right elbow angle
        s12_x, s12_y, s12_vis = extract_landmark_with_visibility(df, 12)
        e14_x, e14_y, e14_vis = extract_landmark_with_visibility(df, 14)
        w16_x, w16_y, w16_vis = extract_landmark_with_visibility(df, 16)
        df['right_elbow_angle'] = calculate_angles_vectorized_2d_with_visibility(
            s12_x, s12_y, s12_vis,
            e14_x, e14_y, e14_vis,
            w16_x, w16_y, w16_vis,
            min_visibility
        )
        
        # Left hip angle
        h23_x, h23_y, h23_vis = extract_landmark_with_visibility(df, 23)
        k25_x, k25_y, k25_vis = extract_landmark_with_visibility(df, 25)
        df['left_hip_angle'] = calculate_angles_vectorized_2d_with_visibility(
            s11_x, s11_y, s11_vis,
            h23_x, h23_y, h23_vis,
            k25_x, k25_y, k25_vis,
            min_visibility
        )
        
        # Right hip angle
        h24_x, h24_y, h24_vis = extract_landmark_with_visibility(df, 24)
        k26_x, k26_y, k26_vis = extract_landmark_with_visibility(df, 26)
        df['right_hip_angle'] = calculate_angles_vectorized_2d_with_visibility(
            s12_x, s12_y, s12_vis,
            h24_x, h24_y, h24_vis,
            k26_x, k26_y, k26_vis,
            min_visibility
        )
        
        # Left spine alignment
        a27_x, a27_y, a27_vis = extract_landmark_with_visibility(df, 27)
        df['left_spine_angle'] = calculate_angles_vectorized_2d_with_visibility(
            s11_x, s11_y, s11_vis,
            h23_x, h23_y, h23_vis,
            a27_x, a27_y, a27_vis,
            min_visibility
        )
        
        # Right spine alignment
        a28_x, a28_y, a28_vis = extract_landmark_with_visibility(df, 28)
        df['right_spine_angle'] = calculate_angles_vectorized_2d_with_visibility(
            s12_x, s12_y, s12_vis,
            h24_x, h24_y, h24_vis,
            a28_x, a28_y, a28_vis,
            min_visibility
        )
    
    # Head tilt (always 2D)
    df['head_tilt_angle'] = calculate_head_tilt_vectorized_with_visibility(df, 0.9)
    
    return df

def process_csv_file(input_path, output_path, use_3d=False, min_visibility=0.5, verbose=True):
    """
    Process a single CSV file to add angle features
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        use_3d: Whether to use 3D angles
        min_visibility: Minimum visibility threshold
        verbose: Whether to print statistics
    
    Returns:
        dict with statistics
    """
    try:
        # Load CSV
        df = pd.read_csv(input_path)
        
        # Add angles
        df_with_angles = add_angle_features(df, use_3d=use_3d, min_visibility=min_visibility)
        
        # Calculate statistics
        angle_columns = [
            'left_elbow_angle', 'right_elbow_angle',
            'left_hip_angle', 'right_hip_angle',
            'left_spine_angle', 'right_spine_angle',
            'head_tilt_angle'
        ]
        
        stats = {
            'filename': input_path.name,
            'total_frames': len(df_with_angles),
            'angles': {}
        }
        
        for col in angle_columns:
            nan_count = df_with_angles[col].isna().sum()
            valid_count = len(df_with_angles) - nan_count
            stats['angles'][col] = {
                'valid': valid_count,
                'nan': nan_count,
                'nan_percentage': (nan_count / len(df_with_angles)) * 100
            }
        
        # Save
        df_with_angles.to_csv(output_path, index=False)
        
        if verbose:
            print(f"✓ Processed: {input_path.name}")
            print(f"  Frames: {stats['total_frames']}")
            total_nans = sum(s['nan'] for s in stats['angles'].values())
            total_values = len(angle_columns) * stats['total_frames']
            print(f"  NaN values: {total_nans}/{total_values} ({(total_nans/total_values)*100:.1f}%)")
        
        return stats
        
    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {str(e)}")
        return None

def process_folder(input_folder, output_folder, use_3d=False, min_visibility=0.5, verbose=True):
    """
    Process all CSV files in a folder
    
    Args:
        input_folder: Path to folder containing input CSVs
        output_folder: Path to folder for output CSVs
        use_3d: Whether to use 3D angles
        min_visibility: Minimum visibility threshold
        verbose: Whether to print progress
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Settings: use_3d={use_3d}, min_visibility={min_visibility}")
    print("="*60)
    
    # Process each file
    all_stats = []
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        output_file = output_path / csv_file.name
        stats = process_csv_file(
            csv_file, 
            output_file, 
            use_3d=use_3d, 
            min_visibility=min_visibility,
            verbose=False  # Disable per-file verbose to avoid cluttering tqdm
        )
        if stats:
            all_stats.append(stats)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed: {len(all_stats)}/{len(csv_files)} files")
    
    if all_stats:
        total_frames = sum(s['total_frames'] for s in all_stats)
        print(f"Total frames processed: {total_frames}")
        
        # Calculate overall NaN statistics
        angle_columns = [
            'left_elbow_angle', 'right_elbow_angle',
            'left_hip_angle', 'right_hip_angle',
            'left_spine_angle', 'right_spine_angle',
            'head_tilt_angle'
        ]
        
        
        print("\nOverall NaN statistics:")
        for col in angle_columns:
            total_nan = sum(s['angles'][col]['nan'] for s in all_stats)
            total_values = sum(s['total_frames'] for s in all_stats)
            pct = (total_nan / total_values) * 100 if total_values > 0 else 0
            print(f"  {col:25} {total_nan:6}/{total_values:6} ({pct:5.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate joint angles from MediaPipe pose landmark CSVs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input folder containing CSV files with pose landmarks'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output folder for CSV files with added angle columns'
    )
    
    parser.add_argument(
        '--3d',
        dest='use_3d',
        action='store_true',
        help='Use 3D angles (x, y, z) instead of 2D (x, y)'
    )
    
    parser.add_argument(
        '--min-visibility',
        type=float,
        default=0.5,
        help='Minimum visibility threshold (0.0 to 1.0). Angles with lower visibility will be NaN'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate min_visibility
    if not 0.0 <= args.min_visibility <= 1.0:
        parser.error("--min-visibility must be between 0.0 and 1.0")
    
    # Process folder
    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        use_3d=args.use_3d,
        min_visibility=args.min_visibility,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()