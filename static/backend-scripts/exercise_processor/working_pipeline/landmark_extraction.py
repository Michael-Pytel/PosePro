import os
from dataclasses import dataclass
from pose_initializing import pose_initialization_processor
import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from video_rotation import detect_video_rotation, rotate_frame

DEBUG = False

@dataclass(frozen=True)
class ChunkSpec:
    start: int
    end: int
    overlap: int

def extract_landmarks_from_frame(pose_landmarks):
    if not pose_landmarks or len(pose_landmarks) == 0:
        return np.array([])

    # Take first pose (we set num_poses=1)
    landmarks = pose_landmarks[0]

    landmarks_array = []
    for landmark in landmarks:
        landmarks_array.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    return np.array(landmarks_array)

def process_frame(landmarker, frame, frame_idx, rotation, fps):
    if rotation != 0:
        frame = rotate_frame(frame, rotation, debug=DEBUG)

    mp_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Mediapipe processing
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_frame_rgb)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, int((frame_idx / fps) * 1000))
    landmarks = extract_landmarks_from_frame(pose_landmarker_result.pose_landmarks)
    return {'frame': frame_idx,
            'time': frame_idx / fps,
            'landmarks': landmarks}

def build_chunks_equal_to_workers(nframes, num_workers, overlap):

    workers = min(num_workers, nframes)
    base = nframes // workers
    remainder = nframes % workers

    chunks = []
    start = 0
    for i in range(workers):
        length = base + (1 if i < remainder else 0)
        end = start + length
        chunks.append(ChunkSpec(start=start, end=end, overlap=overlap))
        start = end
    return chunks

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return nframes, fps

def process_chunk(video_path, chunk, rotation, fps):
    read_from = max(0, chunk.start - chunk.overlap)
    keep_from = chunk.start

    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, read_from)
    results = []
    frame_idx = read_from
    with pose_initialization_processor() as landmarker:
        while frame_idx < chunk.end:
            ret, frame = cap.read()
            if not ret:
                break
            result = process_frame(landmarker, frame, frame_idx, rotation, fps)
            if frame_idx >= keep_from:
                results.append(result)
            frame_idx += 1
    cap.release()
    return results

def extract_landmarks_from_video(video_path):
    num_workers = os.cpu_count() - 1
    rotation = detect_video_rotation(video_path, debug=DEBUG)

    nframes, fps = get_video_metadata(video_path)
    chunks = build_chunks_equal_to_workers(nframes, num_workers=num_workers, overlap=10)

    all_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in chunks:
            fut = executor.submit(
                process_chunk,
                video_path,
                chunk,
                rotation,
                fps,
            )
            futures.append(fut)
        for fut in futures:
            all_results.extend(fut.result())
    return all_results, rotation, fps