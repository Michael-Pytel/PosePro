# -*- coding: utf-8 -*-
import mediapipe
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys
import io

# Set UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


class SuppressOutput:
    """Context manager to suppress all output except what we explicitly allow"""
    def __init__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_stdout = None
        self.save_stderr = None
        
    def __enter__(self):
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)
        sys.stdout = os.fdopen(self.save_stdout, 'w')
        sys.stderr = os.fdopen(self.save_stderr, 'w')
        
    def __exit__(self, *_):
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        os.close(self.null_fd)


def get_model_path(model_complexity='full', script_dir=None):
    """
    Get path to model file in the same directory as script
    
    Args:
        model_complexity: 'lite', 'full', or 'heavy'
        script_dir: Directory where script is located (auto-detected if None)
    
    Returns:
        Path to model file
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    script_dir = os.path.abspath(script_dir)
    model_name = f'pose_landmarker_{model_complexity}.task'

    # Preferred location: ../model relative to this script
    preferred_dir = os.path.abspath(os.path.join(script_dir, '..', 'model'))
    preferred_path = os.path.join(preferred_dir, model_name)

    # Fallback location: same directory as script
    fallback_path = os.path.join(script_dir, model_name)

    if os.path.exists(preferred_path):
        return preferred_path

    if os.path.exists(fallback_path):
        return fallback_path

    # If not found, raise with clear guidance to place the model in ../model
    raise FileNotFoundError(
        f"Model file not found. Checked these locations:\n"
        f"- {preferred_path}\n"
        f"- {fallback_path}\n\n"
        f"Please download the model from:\n"
        f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_complexity}/float16/latest/pose_landmarker_{model_complexity}.task\n"
        f"and place it in: {preferred_dir}"
    )


def initialize_pose(model_path, running_mode='VIDEO'):
    """
    Initialize MediaPipe Pose Landmarker with new API
    
    Args:
        model_path: Path to .task model file
        running_mode: 'IMAGE' or 'VIDEO'
    
    Returns:
        PoseLandmarker object
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    if running_mode == 'VIDEO':
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:  # IMAGE mode
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
    
    with SuppressOutput():
        landmarker = vision.PoseLandmarker.create_from_options(options)
    
    return landmarker


def extract_landmarks_array(pose_landmarks):
    """
    Extract landmarks as numpy array from new API results
    
    Args:
        pose_landmarks: PoseLandmarkerResult pose_landmarks
    
    Returns:
        np.array of shape (33, 4) with [x, y, z, visibility] or None
    """
    if not pose_landmarks or len(pose_landmarks) == 0:
        return None
    
    # Take first pose (we set num_poses=1)
    landmarks = pose_landmarks[0]
    
    landmarks_array = []
    for landmark in landmarks:
        landmarks_array.append([
            landmark.x, 
            landmark.y, 
            landmark.z, 
            landmark.visibility
        ])
    
    return np.array(landmarks_array)


def extract_from_video(video_path, pose_landmarker):
    """
    Extract landmarks from entire video using new API
    
    Args:
        video_path: Path to video file
        pose_landmarker: PoseLandmarker object
    
    Returns:
        list of landmark arrays for each frame, fps
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_landmarks = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame_rgb)
        
        # Process frame with timestamp in milliseconds
        timestamp_ms = frame_number
        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract landmarks as array
        landmarks_array = extract_landmarks_array(results.pose_landmarks)
        
        all_landmarks.append({
            'frame': frame_number,
            'timestamp': frame_number / fps if fps > 0 else frame_number,
            'landmarks': landmarks_array
        })
        
        frame_number += 1
    
    cap.release()
    return all_landmarks, fps


def landmarks_to_dataframe_flat(all_landmarks):
    """
    Convert landmarks to flat DataFrame with all coordinates in single row
    
    Args:
        all_landmarks: List of dicts with frame, timestamp, landmarks
    
    Returns:
        DataFrame with one row per frame, all landmarks flattened
    """
    rows = []
    
    for frame_data in all_landmarks:
        if frame_data['landmarks'] is None:
            continue
        
        row = {
            'frame': frame_data['frame'],
            'timestamp': frame_data['timestamp']
        }
        
        # Flatten all landmarks (33 landmarks × 4 values = 132 features)
        landmarks_flat = frame_data['landmarks'].flatten()
        
        for i, val in enumerate(landmarks_flat):
            row[f'feature_{i}'] = val
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_hash_from_filename(filename):
    """
    Extract hash from filename
    
    Args:
        filename: e.g., 'pushup_athlete0022_vid004_rep08_62865eb4.mp4'
    
    Returns:
        hash string: e.g., '62865eb4'
    """
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split('_')
    hash_value = parts[-1]
    return hash_value


def process_video(video_path, model_path, output_dir='./output', skip_existing=True):
    """
    Complete pipeline: video → landmarks → CSV
    
    Args:
        video_path: Path to video file
        model_path: Path to model file
        output_dir: Directory to save CSV files
        skip_existing: Skip if CSV already exists
    
    Returns:
        tuple: (output_path, status) where status is 'success', 'skipped', or 'failed'
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract hash from filename
    filename = os.path.basename(video_path)
    hash_value = extract_hash_from_filename(filename)
    output_filename = f"{hash_value}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # CHECK IF ALREADY EXISTS
    if skip_existing and os.path.exists(output_path):
        return output_path, 'skipped'
    
    # Create NEW pose landmarker for this video (warnings suppressed inside initialize_pose)
    pose_landmarker = initialize_pose(model_path, running_mode='VIDEO')
    
    try:
        # Extract landmarks from video
        all_landmarks, fps = extract_from_video(video_path, pose_landmarker)
        
        # Convert to DataFrame
        df = landmarks_to_dataframe_flat(all_landmarks)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path, 'success'
    
    except Exception as e:
        return video_path, f'failed: {str(e)}'
    
    finally:
        # Always close the landmarker
        pose_landmarker.close()


def process_video_wrapper(args):
    """
    Wrapper function for multiprocessing
    
    Args:
        args: tuple of (video_path, model_path, output_dir, skip_existing)
    
    Returns:
        tuple: (video_path, output_path, status)
    """
    video_path, model_path, output_dir, skip_existing = args
    output_path, status = process_video(video_path, model_path, output_dir, skip_existing)
    return video_path, output_path, status


def process_folder(input_folder, output_dir='./output', model_complexity='full', 
                  video_extensions=None, skip_existing=True, num_workers=None):
    """
    Process all videos in a folder with optional multiprocessing
    
    Args:
        input_folder: Path to folder containing videos
        output_dir: Directory to save CSV files
        model_complexity: 'lite', 'full', or 'heavy'
        video_extensions: List of video extensions to process
        skip_existing: Skip videos that already have CSV files
        num_workers: Number of parallel workers (None = auto-detect, 1 = no multiprocessing)
    
    Returns:
        List of paths to saved CSV files
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Get model path from script directory
    model_path = get_model_path(model_complexity)
    
    # Find all files in folder
    all_files = glob.glob(os.path.join(input_folder, '*'))
    
    # Filter by extension (case-insensitive)
    video_files = []
    for file_path in all_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [ext.lower() for ext in video_extensions]:
            video_files.append(file_path)
    
    video_files.sort()
    
    if not video_files:
        print(f"ERROR: No video files found in {input_folder}")
        print(f"Looking for extensions: {video_extensions}")
        return []
    
    # Auto-detect number of workers if not specified
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Determine if using multiprocessing
    use_multiprocessing = num_workers > 1
    
    print(f"\n{'='*70}")
    print(f"FOLDER PROCESSING")
    print(f"{'='*70}")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Model:         {model_complexity}")
    print(f"Videos found:  {len(video_files)}")
    print(f"Workers:       {num_workers} {'(multiprocessing)' if use_multiprocessing else '(single process)'}")
    print(f"Mode:          {'RESUME (skip existing)' if skip_existing else 'REPROCESS ALL'}")
    print(f"{'='*70}\n")
    
    # Prepare arguments for processing
    args_list = [
        (video_path, model_path, output_dir, skip_existing)
        for video_path in video_files
    ]
    
    # Process videos
    output_paths = []
    successful = 0
    failed = 0
    skipped = 0
    
    if use_multiprocessing:
        print(f"Processing {len(video_files)} videos with {num_workers} workers...\n")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Use tqdm to show progress
            results = list(tqdm(
                executor.map(process_video_wrapper, args_list),
                total=len(video_files),
                desc="Progress",
                unit="video",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ))
        
        # Count results
        for video_path, output_path, status in results:
            if status == 'success':
                output_paths.append(output_path)
                successful += 1
            elif status == 'skipped':
                output_paths.append(output_path)
                skipped += 1
            else:  # failed
                print(f"\n❌ ERROR: {os.path.basename(video_path)} - {status}")
                failed += 1
    
    else:
        print(f"Processing {len(video_files)} videos sequentially...\n")
        
        for i, (video_path, model_path_arg, output_dir_arg, skip_existing_arg) in enumerate(args_list, 1):
            filename = os.path.basename(video_path)
            
            # Check before processing if file will be skipped
            hash_value = extract_hash_from_filename(filename)
            output_path = os.path.join(output_dir, f"{hash_value}.csv")
            
            will_skip = skip_existing and os.path.exists(output_path)
            
            if will_skip:
                print(f"[{i}/{len(video_files)}] ⏭️  SKIP: {filename}")
                output_paths.append(output_path)
                skipped += 1
            else:
                print(f"[{i}/{len(video_files)}] 🔄 Processing: {filename}...", end='', flush=True)
                
                try:
                    # Create NEW pose landmarker for this video
                    pose_landmarker = initialize_pose(model_path, running_mode='VIDEO')
                    
                    try:
                        # Extract landmarks
                        all_landmarks, fps = extract_from_video(video_path, pose_landmarker)
                        
                        # Convert to DataFrame
                        df = landmarks_to_dataframe_flat(all_landmarks)
                        
                        # Save to CSV
                        df.to_csv(output_path, index=False)
                        
                        print(f" ✓ ({len(df)} frames)")
                        
                        output_paths.append(output_path)
                        successful += 1
                    
                    finally:
                        pose_landmarker.close()
                
                except Exception as e:
                    print(f" ❌ ERROR: {str(e)}")
                    failed += 1
                    continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total videos:  {len(video_files)}")
    print(f"✓ Successful:  {successful}")
    print(f"⏭️  Skipped:     {skipped}")
    print(f"✗ Failed:      {failed}")
    print(f"Output:        {output_dir}")
    print(f"{'='*70}\n")
    
    return output_paths


def main():
    """
    Main function for CLI
    """
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe landmarks from videos and save as CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_landmarks.py -i ./videos -o ./output
  python extract_landmarks.py -i ./videos -o ./output --model heavy
  python extract_landmarks.py -i ./videos -o ./output --workers 4
  python extract_landmarks.py -i ./videos -o ./output --no-skip

Model files must be in the same directory as this script:
  - pose_landmarker_lite.task
  - pose_landmarker_full.task
  - pose_landmarker_heavy.task

Download from:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='./media/athlete_videos/videos_nooverlay',
        help='Input folder containing video files (default: ./media/athlete_videos/videos_nooverlay)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./media/athlete_videos/output',
        help='Output folder to save CSV files (default: ./media/athlete_videos/output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['lite', 'full', 'heavy'],
        default='full',
        help='Model complexity (default: full)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.mp4', '.avi', '.mov', '.mkv'],
        help='Video file extensions to process (default: .mp4 .avi .mov .mkv)'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Reprocess all videos (do not skip existing CSVs)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1). Use 1 to disable multiprocessing.'
    )
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input folder does not exist: {args.input}")
        return
    
    if not os.path.isdir(args.input):
        print(f"ERROR: Input path is not a folder: {args.input}")
        return
    
    # Validate workers
    if args.workers is not None and args.workers < 1:
        print(f"ERROR: --workers must be at least 1")
        return
    
    # Process folder
    try:
        process_folder(
            input_folder=args.input,
            output_dir=args.output,
            model_complexity=args.model,
            video_extensions=args.extensions,
            skip_existing=not args.no_skip,
            num_workers=args.workers
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return


if __name__ == "__main__":
    main()