import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe_extract import extract_landmarks as el
from compute_signals import compute_pushup_signals as comp
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import pandas as pd
import re

@dataclass
class RepetitionRecord:
    """Rekord pojedynczego powtórzenia do CSV"""
    # Identyfikatory
    person_id: str
    exercise_type: str  # 'pushup' lub 'squat'
    video_id: str
    rep_number: int
    unique_rep_id: str
    video_filename: str
    
    # Parametry czasowe
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    
    # Metryki jakości
    consensus_score: float
    quality_score: float
    
    # Szczegóły techniczne
    signal_used: str
    is_edge_rep: bool
    fps: float
    
    # Metryki ruchu (uniwersalne)
    primary_joint_range: float  # biodra dla obu
    primary_angle_mean: float  # łokcie dla pompek, kolana dla przysiadów
    primary_angle_std: float
    secondary_angle_mean: float  # tułów dla obu
    symmetry_score: float
    
    # Specyficzne dla przysiadów
    depth_category: Optional[str]  # 'deep', 'parallel', 'shallow' lub None dla pompek
    min_knee_angle: Optional[float]  # tylko dla przysiadów
    max_heel_lift: Optional[float]  # maksymalne oderwanie pięty od podłoża (tylko przysiady)
    
    # Ścieżki do plików
    video_clip_path: str
    visualization_path: str
    
    # Timestamp
    processing_timestamp: str



class ExerciseBatchProcessor:
    """
    Uniwersalny pipeline do przetwarzania pompek i przysiadów
    """
    
    def __init__(
            self,
            exercise_type: str = 'pushup',
            min_rep_duration: float = None,
            max_rep_duration: float = None,
            peak_prominence: float = 0.08,
            video_quality: str = 'medium',
            debug: bool = False,
            visualize_landmarks: bool = False  # ⭐ DODANE
        ):
        if exercise_type not in ['pushup', 'squat']:
            raise ValueError("exercise_type musi być 'pushup' lub 'squat'")
        
        self.exercise_type = exercise_type
        
        # Domyślne wartości zależne od typu ćwiczenia
        if min_rep_duration is None:
            self.min_rep_duration = 0.6 if exercise_type == 'pushup' else 0.7
        else:
            self.min_rep_duration = min_rep_duration
            
        if max_rep_duration is None:
            self.max_rep_duration = 6.0 if exercise_type == 'pushup' else 8.0
        else:
            self.max_rep_duration = max_rep_duration
        
        self.peak_prominence = peak_prominence
        self.video_quality = video_quality
        self.debug = debug
        
        # MediaPipe - podkręcony (jak w dedykowanym procesorze)
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(
        #     static_image_mode=False,
        #     model_complexity=2,
        #     smooth_landmarks=True,
        #     enable_segmentation=True,
        #     min_detection_confidence=0.7,
        #     min_tracking_confidence=0.7
        # )
        
        self.has_ffmpeg = self._check_ffmpeg()
        
        # ⭐ DODANE - Zapisz flagę wizualizacji
        self.visualize_landmarks = visualize_landmarks
        
        # ⭐ DODANE - Inicjalizacja MediaPipe Drawing (tylko jeśli wizualizacja)
        if self.visualize_landmarks:
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Licznik globalny dla unique ID
        self.global_rep_counter = 0
    
    def _check_ffmpeg(self) -> bool:
        """Sprawdza dostępność ffmpeg"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except:
            return False

    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        person_id: str = "unknown",
        video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.MOV']
    ) -> pd.DataFrame:
        """
        Przetwarza cały folder z nagraniami
        
        Args:
            input_folder: Folder z nagraniami wejściowymi
            output_folder: Folder docelowy
            person_id: ID osoby (np. "person123", "athlete_001")
            video_extensions: Rozszerzenia plików wideo do przetworzenia
            
        Returns:
            DataFrame z wszystkimi powtórzeniami
        """
        exercise_name = "POMPEK" if self.exercise_type == 'pushup' else "PRZYSIADÓW"
        
        print("\n" + "="*80)
        print(f"🚀 BATCH PROCESSING - SEGMENTACJA {exercise_name}")
        print("="*80)
        print(f"📁 Folder wejściowy: {input_folder}")
        print(f"📂 Folder wyjściowy: {output_folder}")
        print(f"👤 Person ID: {person_id}")
        print(f"🏋️ Typ ćwiczenia: {self.exercise_type}")
        print("="*80 + "\n")
        
        # Stwórz strukturę folderów
        output_path = Path(output_folder)
        videos_dir = output_path / "videos"
        viz_dir = output_path / "visualizations"
        videos_nooverlay_dir = output_path / "videos_nooverlay"
        videos_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        videos_nooverlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Znajdź wszystkie pliki wideo
        input_path = Path(input_folder)
        video_files = []
        seen = set()
        
        for ext in video_extensions:
            for vf in input_path.glob(f"*{ext}"):
                normalized = str(vf).lower()
                if normalized not in seen:
                    seen.add(normalized)
                    video_files.append(vf)
        
        video_files = sorted(video_files)
        
        if not video_files:
            print(f"❌ Nie znaleziono plików wideo w: {input_folder}")
            print(f"   Szukane rozszerzenia: {video_extensions}")
            return pd.DataFrame()
        
        print(f"✅ Znaleziono {len(video_files)} plików wideo\n")
        
        # Lista wszystkich rekordów
        all_records = []
        
        # Przetwarzaj każde wideo
        for video_idx, video_file in enumerate(video_files, start=1):
            video_id = f"vid{video_idx:03d}"

            print(f"\n{'=' * 80}")
            print(f"📹 [{video_idx}/{len(video_files)}] Przetwarzam: {video_file.name}")
            print(f"   Video ID: {video_id}")
            print(f"{'=' * 80}")

            try:
                # Przetwórz pojedyncze wideo
                video_records = self._process_single_video(
                    video_path=str(video_file),
                    person_id=person_id,
                    video_id=video_id,
                    videos_dir=videos_dir,
                    videos_nooverlay_dir=videos_nooverlay_dir,
                    viz_dir=viz_dir,
                    video_filename=video_file.name
                )

                all_records.extend(video_records)
                print(f"✅ Przetworzono: {len(video_records)} powtórzeń z {video_file.name}")

            except Exception as e:
                print(f"❌ Błąd przetwarzania {video_file.name}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue
        # Stwórz DataFrame
        if all_records:
            df = pd.DataFrame([asdict(r) for r in all_records])
            
            # Zapisz CSV
            csv_path = output_path / "repetitions_data.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"\n{'='*80}")
            print("PODSUMOWANIE")
            print(f"{'='*80}")
            print(f"Przetworzone wideo: {len(video_files)}")
            print(f"Wykryte powtórzenia: {len(all_records)}")
            print(f"CSV zapisany: {csv_path}")
            print(f"Klipy wideo: {videos_dir}")
            print(f"Wizualizacje: {viz_dir}")
            
            # Statystyki
            print(f"\nStatystyki jakości:")
            print(f"   Średni consensus score: {df['consensus_score'].mean():.3f}")
            print(f"   Średni quality score: {df['quality_score'].mean():.3f}")
            print(f"   Średni czas trwania: {df['duration'].mean():.2f}s")
            print(f"   Powtórzenia brzegowe: {df['is_edge_rep'].sum()}")
            
            # Statystyki specyficzne dla przysiadów
            if self.exercise_type == 'squat':
                print(f"\n  Statystyki głębokości przysiadów:")
                depth_counts = df['depth_category'].value_counts()
                for depth, count in depth_counts.items():
                    print(f"   • {depth}: {count}")
            
            print(f"{'='*80}\n")
            
            return df
        else:
            print("\n Nie udało się przetworzyć żadnych powtórzeń")
            return pd.DataFrame()
    
    def _process_single_video(
        self,
        video_path: str,
        person_id: str,
        video_id: str,
        videos_dir: Path,
        videos_nooverlay_dir: Path,
        viz_dir: Path,
        video_filename: str
    ) -> List[RepetitionRecord]:
        """Przetwarza pojedyncze wideo i zwraca rekordy powtórzeń"""
        self._current_video_path = video_path

        # 1. Ekstrakcja landmarks
        print("📍 Ekstrakcja punktów ciała...")
        landmarks_data = self._extract_landmarks(video_path)

        if self.visualize_landmarks:
            print("🎨 Tworzenie pełnego wideo z wizualizacją landmarks...")
            full_viz_path = videos_dir / f"{person_id}_{self.exercise_type}_{video_id}_FULL_visualization.mp4"
            self._create_full_video_with_landmarks(
                video_path=video_path,
                landmarks_data=landmarks_data,
                output_path=full_viz_path
            )
            print(f"✅ Zapisano: {full_viz_path.name}")

        if not landmarks_data:
            print("❌ Nie udało się wyekstrahować landmarks")
            return []

        print(f"✅ Wyekstrahowano {len(landmarks_data)} klatek z poprawnymi landmarkami")

        # 2. Oblicz sygnały (zależnie od typu ćwiczenia)
        print("📊 Obliczam sygnały ruchu...")
        if self.exercise_type == 'pushup':
            signals = comp(landmarks_data, self.fps)
        else:  # squat
            signals = self._compute_squat_signals(landmarks_data)

        # 3. Detekcja powtórzeń (PEAK–VALLEY–PEAK / VALLEY–PEAK–VALLEY)
        print("🔍 Detekcja powtórzeń...")
        if self.exercise_type == 'pushup':
            repetitions = self._detect_pushup_repetitions(signals, landmarks_data)
        else:  # squat
            repetitions = self._detect_squat_repetitions(signals, landmarks_data)

        if not repetitions:
            print("⚠️  Nie wykryto powtórzeń")
            return []

        # 4. Walidacja jakości
        print("🎯 Walidacja jakości...")
        if self.exercise_type == 'pushup':
            validated_reps = self._validate_pushup_repetitions(repetitions, landmarks_data)
        else:  # squat
            validated_reps = self._validate_squat_repetitions(repetitions, landmarks_data)

        print(f"✅ Zatwierdzono: {len(validated_reps)} powtórzeń")

        # 5. Wizualizacja sygnałów (pracuje w przestrzeni indeksów landmarks, więc zostawiamy stare start_time/end_time)
        viz_filename = f"{person_id}_{self.exercise_type}_{video_id}.png"
        viz_path = viz_dir / viz_filename
        self._create_visualization(signals, validated_reps, viz_path, person_id, video_id)

        # 6. Wytnij klipy i stwórz rekordy
        records = []
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps if self.fps and self.fps > 0 else cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # bezpieczny fallback

        n_landmarks = len(landmarks_data)

        for rep in validated_reps:
            rep_number = rep['rep_id']

            # 1) indeksy w przestrzeni sygnałów / landmarks
            start_idx = int(rep['start_frame'])
            end_idx = int(rep['end_frame'])

            # zabezpieczenie zakresów
            start_idx = max(0, min(start_idx, n_landmarks - 1))
            end_idx = max(0, min(end_idx, n_landmarks - 1))

            # 2) prawdziwe numery klatek w oryginalnym wideo
            start_frame = int(landmarks_data[start_idx]['frame'])
            end_frame = int(landmarks_data[end_idx]['frame'])

            if end_frame < start_frame:
                end_frame = start_frame

            # 3) czasy w sekundach w przestrzeni WIDEO (używane w CSV i przy ffmpeg)
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = max(0.0, end_time - start_time)

            # Generuj unique ID
            unique_id = self._generate_unique_id(person_id, video_id, rep_number)
            self.global_rep_counter += 1

            # Nazwa pliku
            if self.exercise_type == 'squat' and 'depth_category' in rep:
                video_clip_filename = (
                    f"{self.exercise_type}_{person_id}_{video_id}_rep{rep_number:02d}_"
                    f"{rep['depth_category']}_{unique_id}.mp4"
                )
            else:
                video_clip_filename = (
                    f"{self.exercise_type}_{person_id}_{video_id}_rep{rep_number:02d}_{unique_id}.mp4"
                )

            video_clip_path = videos_dir / video_clip_filename

            # 4) Wytnij klip – teraz po PRAWDZIWYCH numerach klatek
            if self.visualize_landmarks:
                # Tryb z wizualizacją landmarks
                self._extract_video_clip_with_visualization(
                    video_path=video_path,
                    landmarks_data=landmarks_data,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    output_path=video_clip_path,
                    width=width,
                    height=height,
                    fps=fps,
                    rep_info={
                        'rep_number': rep_number,
                        'quality_score': rep.get('quality_score', 0.0),
                        'depth_category': rep.get('depth_category'),
                        'min_knee_angle': rep.get('min_knee_angle'),
                        'consensus_score': rep.get('consensus_score', 0.0)
                    }
                )
                # Additionally save a raw (no-overlay) version of the clip
                try:
                    # Save raw clip using the same filename (no extra suffix)
                    raw_video_clip_path = videos_nooverlay_dir / video_clip_filename
                    # Use the ffmpeg path when available (uses self._current_video_path),
                    # otherwise fallback will read frames from the provided cap.
                    self._extract_video_clip(
                        cap,
                        start_frame,
                        end_frame,
                        raw_video_clip_path,
                        width,
                        height,
                        fps
                    )
                    if self.debug:
                        print(f"✅ Zapisano surowy klip bez nakładki: {raw_video_clip_path.name}")
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ Nie udało się zapisać surowego klipu: {e}")
            else:
                # Tryb standardowy (szybki, bez wizualizacji)
                self._extract_video_clip(
                    cap, start_frame, end_frame,
                    video_clip_path, width, height, fps
                )

            # 5) Metryki jakości liczymy dalej w przestrzeni indeksów landmarks
            if self.exercise_type == 'pushup':
                metrics = self._calculate_pushup_metrics(
                    landmarks_data[start_idx:end_idx + 1]
                )
            else:  # squat
                metrics = self._calculate_squat_metrics(
                    landmarks_data[start_idx:end_idx + 1]
                )

            # 6) Rekord do CSV – zapisujemy już PRAWDZIWE czasy i klatki
            record = RepetitionRecord(
                person_id=person_id,
                exercise_type=self.exercise_type,
                video_id=video_id,
                rep_number=rep_number,
                unique_rep_id=unique_id,
                video_filename=video_filename,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                consensus_score=rep.get('consensus_score', 0.0),
                quality_score=rep.get('quality_score', 0.0),
                signal_used=rep.get('signal_used', 'unknown'),
                is_edge_rep=rep.get('note') in ['edge_start', 'edge_end'],
                fps=fps,
                primary_joint_range=metrics['primary_joint_range'],
                primary_angle_mean=metrics['primary_angle_mean'],
                primary_angle_std=metrics['primary_angle_std'],
                secondary_angle_mean=metrics['secondary_angle_mean'],
                symmetry_score=metrics['symmetry_score'],
                depth_category=rep.get('depth_category') if self.exercise_type == 'squat' else None,
                min_knee_angle=rep.get('min_knee_angle') if self.exercise_type == 'squat' else None,
                max_heel_lift=rep.get('max_heel_lift') if self.exercise_type == 'squat' else None,
                video_clip_path=str(video_clip_path.relative_to(videos_dir.parent)),
                visualization_path=str(viz_path.relative_to(viz_dir.parent)),
                processing_timestamp=datetime.now().isoformat()
            )

            records.append(record)

        cap.release()
        return records

    
    def _generate_unique_id(self, person_id: str, video_id: str, rep_number: int) -> str:
        """Generuje unikalne ID dla powtórzenia"""
        content = f"{person_id}_{video_id}_{rep_number}_{datetime.now().isoformat()}_{self.global_rep_counter}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()[:8]
    
    def _extract_landmarks(self, video_path: str) -> List[Dict]:
        """
        Ekstrakcja landmarks z wideo
        """
        MODEL_PATH = Path(
            r"C:\Users\micha\Documents\camera-based-exercise-evaluation\django-app\static\backend-scripts\exercise_processor\model\pose_landmarker_full.task"
        ).resolve()
        print("MODEL PATH:", MODEL_PATH, MODEL_PATH.exists())  # DEBUG!

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO)

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        with PoseLandmarker.create_from_options(options) as landmarker:
            test = el(landmarker, video_path)
            return test
        # cap = cv2.VideoCapture(video_path)
        # self.fps = cap.get(cv2.CAP_PROP_FPS)
        #
        # # ⭐ Wyłącz automatyczną rotację w OpenCV (jeśli dostępne)
        # # Niektóre wersje OpenCV mają CAP_PROP_ORIENTATION_AUTO
        # try:
        #     cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        # except:
        #     pass
        #
        # # Wykryj rotację z metadanych (tylko dla MediaPipe, nie dla outputu!)
        # rotation = self._detect_video_rotation(video_path)
        #
        # landmarks_data = []
        # frame_idx = 0
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #
        # # Statystyki detekcji
        # detection_stats = {
        #     'total_frames': 0,
        #     'detected_frames': 0,
        #     'failed_frames': 0
        # }
        #
        # while cap.isOpened():
        #     ret, frame = cap.read()  # ⭐ Oryginalna klatka - NIE ZMIENIAMY!
        #     if not ret:
        #         break
        #
        #     detection_stats['total_frames'] += 1
        #
        #     # ⭐ KLUCZOWE: Tworzymy KOPIĘ dla MediaPipe, oryginał pozostaje nietknięty!
        #     frame_for_mediapipe = frame.copy()
        #
        #     # KROK 1: Korekcja rotacji TYLKO dla MediaPipe
        #     if rotation != 0:
        #         frame_for_mediapipe = self._rotate_frame(frame_for_mediapipe, rotation)
        #
        #     # KROK 2: Poprawa jakości dla MediaPipe
        #     frame_enhanced = cv2.convertScaleAbs(frame_for_mediapipe, alpha=1.3, beta=15)
        #
        #     # KROK 3: Konwersja do RGB
        #     frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
        #
        #     # KROK 4: Przetwórz przez MediaPipe
        #     results = self.pose.process(frame_rgb)
        #
        #     # KROK 5: Walidacja jakości detekcji
        #     if results.pose_landmarks:
        #         # Sprawdź czy mamy kluczowe punkty dla pompek
        #         key_points = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28]  # Ramiona, łokcie, nadgarstki, biodra, kostki
        #         visible_key_points = 0
        #
        #         landmarks = {}
        #         for idx, lm in enumerate(results.pose_landmarks.landmark):
        #             landmarks[idx] = {
        #                 'x': lm.x,
        #                 'y': lm.y,
        #                 'z': lm.z,
        #                 'visibility': lm.visibility
        #             }
        #
        #             # Zlicz widoczne kluczowe punkty
        #             if idx in key_points and lm.visibility > 0.5:
        #                 visible_key_points += 1
        #
        #         # Akceptuj tylko jeśli mamy co najmniej 70% kluczowych punktów
        #         if visible_key_points >= len(key_points) * 0.7:
        #             landmarks_data.append({
        #                 'frame': frame_idx,
        #                 'time': frame_idx / self.fps,
        #                 'landmarks': landmarks,
        #                 'visibility_score': visible_key_points / len(key_points),
        #                 'rotation_applied': rotation  # INFO: rotacja była stosowana tylko do MediaPipe
        #             })
        #             detection_stats['detected_frames'] += 1
        #         else:
        #             detection_stats['failed_frames'] += 1
        #     else:
        #         detection_stats['failed_frames'] += 1
        #
        #     frame_idx += 1
        #
        #     if frame_idx % 50 == 0 and self.debug:
        #         progress = (frame_idx / total_frames) * 100
        #         detection_rate = (detection_stats['detected_frames'] / detection_stats['total_frames']) * 100
        #         print(f"  Postęp: {progress:.1f}% | Detekcja: {detection_rate:.1f}%", end='\r')
        #
        # cap.release()
        #
        # # Raport detekcji
        # if detection_stats['total_frames'] > 0:
        #     detection_rate = (detection_stats['detected_frames'] / detection_stats['total_frames']) * 100
        #     print(f"\n  📊 Detekcja: {detection_stats['detected_frames']}/{detection_stats['total_frames']} "
        #           f"({detection_rate:.1f}%)")
        #
        #     if rotation != 0:
        #         print(f"  🔄 Rotacja {rotation}° stosowana TYLKO dla MediaPipe (output bez rotacji)")
        #
        #     if detection_rate < 50:
        #         print(f"  ⚠️  UWAGA: Niska jakość detekcji ({detection_rate:.1f}%)!")


        return landmarks_data

    def _detect_video_rotation(self, video_path: str) -> int:
        """Detecting video rotation from metadata using ffprobe"""
        try:
            # Using ffprobe to get rotation metadata
            cmd_rotate = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream_tags=rotate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result_rotate = subprocess.run(cmd_rotate, capture_output=True, text=True)

            if result_rotate.returncode == 0:
                txt = result_rotate.stdout.strip()
                if txt and txt.isdigit():
                    rotation = int(txt)
                    if self.debug:
                        print("Detected rotate tag:", rotation)
                    return rotation

            # Alternative: check display_matrix for rotation
            cmd_matrix = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'side_data=rotation',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result_matrix = subprocess.run(cmd_matrix, capture_output=True, text=True)

            if result_matrix.returncode == 0:
                txt = result_matrix.stdout.strip()
                if txt:
                    match = re.search(r'-?\d+', txt)
                    if match:
                        rotation = abs(int(float(match.group())))
                        if self.debug:
                            print("Detected rotation from display matrix:", rotation)
                        return rotation

            # No rotation found
            if self.debug:
                print("No rotation metadata found")
            return 0

        except Exception as e:
            if self.debug:
                print(f"Error detecting rotation: {e}")
            return 0

    
    def _rotate_frame(self, frame: np.ndarray, rotation: int) -> np.ndarray:
        """Obróć klatkę według kąta rotacji"""
        if rotation == 0:
            return frame
        elif rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Dla niestandardowych kątów
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            return cv2.warpAffine(frame, M, (w, h))
    
    def _detect_camera_orientation(self, landmarks_data: List[Dict]) -> str:
        """
        Wykryj orientację kamery na podstawie pierwszych 30 klatek
        
        Returns:
            'side' - widok z boku (najlepszy)
            'front' - widok z przodu
            'angle' - widok pod kątem
            'top' - widok z góry
        """
        if not landmarks_data or len(landmarks_data) < 10:
            return 'unknown'
        
        # Weź próbkę klatek
        sample_size = min(30, len(landmarks_data))
        sample_landmarks = landmarks_data[:sample_size]
        
        shoulder_widths = []  # Szerokość ramion w X
        hip_widths = []  # Szerokość bioder w X
        body_depths = []  # Głębokość ciała w Z
        torso_x_ranges = []  # Zakres ruchu tułowia w X
        torso_y_ranges = []  # Zakres ruchu tułowia w Y
        
        for data in sample_landmarks:
            lm = data['landmarks']
            
            # Szerokość ramion
            if 11 in lm and 12 in lm:
                shoulder_width = abs(lm[11]['x'] - lm[12]['x'])
                shoulder_widths.append(shoulder_width)
            
            # Szerokość bioder
            if 23 in lm and 24 in lm:
                hip_width = abs(lm[23]['x'] - lm[24]['x'])
                hip_widths.append(hip_width)
            
            # Głębokość Z
            if 11 in lm and 12 in lm:
                body_depth = abs(lm[11]['z'] - lm[12]['z'])
                body_depths.append(body_depth)
            
            # Zakres ruchu tułowia
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                torso_center_x = (lm[11]['x'] + lm[12]['x'] + lm[23]['x'] + lm[24]['x']) / 4
                torso_center_y = (lm[11]['y'] + lm[12]['y'] + lm[23]['y'] + lm[24]['y']) / 4
                torso_x_ranges.append(torso_center_x)
                torso_y_ranges.append(torso_center_y)
        
        # Oblicz średnie
        avg_shoulder_width = np.mean(shoulder_widths) if shoulder_widths else 0
        avg_hip_width = np.mean(hip_widths) if hip_widths else 0
        avg_body_depth = np.mean(body_depths) if body_depths else 0
        x_variance = np.var(torso_x_ranges) if torso_x_ranges else 0
        y_variance = np.var(torso_y_ranges) if torso_y_ranges else 0
        
        if self.debug:
            print(f"\n🎥 DETEKCJA ORIENTACJI KAMERY:")
            print(f"   Szerokość ramion: {avg_shoulder_width:.3f}")
            print(f"   Szerokość bioder: {avg_hip_width:.3f}")
            print(f"   Głębokość ciała (Z): {avg_body_depth:.3f}")
            print(f"   Wariancja X: {x_variance:.4f}")
            print(f"   Wariancja Y: {y_variance:.4f}")
        
        # Logika detekcji
        # WIDOK Z BOKU: wąskie ramiona/biodra (< 0.15), duży ruch w Y
        if avg_shoulder_width < 0.15 and avg_hip_width < 0.15 and y_variance > 0.001:
            orientation = 'side'
        # WIDOK Z PRZODU: szerokie ramiona/biodra (> 0.25), mały ruch w X i Y
        elif avg_shoulder_width > 0.25 and avg_hip_width > 0.20:
            orientation = 'front'
        # WIDOK Z GÓRY: duża głębokość Z, mały ruch w Y
        elif avg_body_depth > 0.15 and y_variance < 0.001:
            orientation = 'top'
        # WIDOK POD KĄTEM: wszystko inne
        else:
            orientation = 'angle'
        
        if self.debug:
            print(f"   ➡️  Wykryta orientacja: {orientation.upper()}")
        
        return orientation
    
    def _compute_pushup_signals(self, landmarks_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Oblicz sygnały ruchu dla pompek 
        - Automatyczna detekcja orientacji kamery
        - Wykorzystanie wszystkich dostępnych punktów (33 landmarks MediaPipe)
        - Sygnały 3D (X, Y, Z) zamiast tylko Y
        - Dystanse euklidesowe między kluczowymi punktami
        - Wielowymiarowa analiza ruchu
        """
        
        # Wykryj orientację kamery
        camera_orientation = self._detect_camera_orientation(landmarks_data)
        
        signals = {}
        
        # ========== PODSTAWOWE WYSOKOŚCI (Y) ==========
        avg_hip_y = []
        avg_shoulder_y = []
        avg_wrist_y = []
        avg_elbow_y = []
        avg_knee_y = []
        nose_y = []
        
        # ========== POZYCJE X (bok-bok) ==========
        avg_hip_x = []
        avg_shoulder_x = []
        torso_center_x = []
        
        # ========== POZYCJE Z (głębokość) ==========
        avg_hip_z = []
        avg_shoulder_z = []
        torso_center_z = []
        
        # ========== KĄTY STAWÓW ==========
        left_elbow_angles = []
        right_elbow_angles = []
        left_shoulder_angles = []
        right_shoulder_angles = []
        left_wrist_angles = []
        right_wrist_angles = []
        
        # ========== KĄTY TUŁOWIA ==========
        torso_angles = []
        body_angles = []
        plank_angles = []  # Kąt "deski" - wyprostowanie ciała
        
        # ========== DYSTANSE 3D ==========
        shoulder_hip_distances = []  # Dystans ramiona-biodra
        elbow_knee_distances = []  # Dystans łokcie-kolana
        wrist_shoulder_distances = []  # Dystans nadgarstek-ramię
        chest_ground_distances = []  # Wysokość klatki nad podłożem
        
        # ========== SZEROKOŚCI ==========
        shoulder_widths = []
        hip_widths = []
        elbow_widths = []
        
        # ========== SYMETRIA ==========
        left_right_shoulder_diff = []
        left_right_hip_diff = []
        
        for data in landmarks_data:
            lm = data['landmarks']
            
            # ===== WYSOKOŚCI Y =====
            if 23 in lm and 24 in lm:
                avg_hip_y.append(1 -((lm[23]['y'] + lm[24]['y']) / 2))
                avg_hip_x.append((lm[23]['x'] + lm[24]['x']) / 2)
                avg_hip_z.append((lm[23]['z'] + lm[24]['z']) / 2)
                hip_widths.append(abs(lm[23]['x'] - lm[24]['x']))
                left_right_hip_diff.append(lm[23]['y'] - lm[24]['y'])
            else:
                avg_hip_y.append(np.nan)
                avg_hip_x.append(np.nan)
                avg_hip_z.append(np.nan)
                hip_widths.append(np.nan)
                left_right_hip_diff.append(np.nan)
            
            if 11 in lm and 12 in lm:
                avg_shoulder_y.append(1 - ((lm[11]['y'] + lm[12]['y']) / 2))
                avg_shoulder_x.append((lm[11]['x'] + lm[12]['x']) / 2)
                avg_shoulder_z.append((lm[11]['z'] + lm[12]['z']) / 2)
                shoulder_widths.append(abs(lm[11]['x'] - lm[12]['x']))
                left_right_shoulder_diff.append(lm[11]['y'] - lm[12]['y'])
            else:
                avg_shoulder_y.append(np.nan)
                avg_shoulder_x.append(np.nan)
                avg_shoulder_z.append(np.nan)
                shoulder_widths.append(np.nan)
                left_right_shoulder_diff.append(np.nan)
            
            if 15 in lm and 16 in lm:
                avg_wrist_y.append((lm[15]['y'] + lm[16]['y']) / 2)
            else:
                avg_wrist_y.append(np.nan)
            
            if 13 in lm and 14 in lm:
                avg_elbow_y.append((lm[13]['y'] + lm[14]['y']) / 2)
                elbow_widths.append(abs(lm[13]['x'] - lm[14]['x']))
            else:
                avg_elbow_y.append(np.nan)
                elbow_widths.append(np.nan)
            
            if 25 in lm and 26 in lm:
                avg_knee_y.append((lm[25]['y'] + lm[26]['y']) / 2)
            else:
                avg_knee_y.append(np.nan)
            
            if 0 in lm:  # Nos
                nose_y.append(-lm[0]['y'])
            else:
                nose_y.append(np.nan)
            
            # ===== CENTRUM TUŁOWIA =====
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                torso_center_x.append((lm[11]['x'] + lm[12]['x'] + lm[23]['x'] + lm[24]['x']) / 4)
                torso_center_z.append((lm[11]['z'] + lm[12]['z'] + lm[23]['z'] + lm[24]['z']) / 4)
            else:
                torso_center_x.append(np.nan)
                torso_center_z.append(np.nan)
            
            # ===== KĄTY ŁOKCI =====
            if 11 in lm and 13 in lm and 15 in lm:
                left_elbow_angles.append(self._calculate_angle(lm[11], lm[13], lm[15]))
            else:
                left_elbow_angles.append(np.nan)
            
            if 12 in lm and 14 in lm and 16 in lm:
                right_elbow_angles.append(self._calculate_angle(lm[12], lm[14], lm[16]))
            else:
                right_elbow_angles.append(np.nan)
            
            # ===== KĄTY RAMION =====
            if 13 in lm and 11 in lm and 23 in lm:
                left_shoulder_angles.append(self._calculate_angle(lm[13], lm[11], lm[23]))
            else:
                left_shoulder_angles.append(np.nan)
            
            if 14 in lm and 12 in lm and 24 in lm:
                right_shoulder_angles.append(self._calculate_angle(lm[14], lm[12], lm[24]))
            else:
                right_shoulder_angles.append(np.nan)
            
            # ===== KĄTY NADGARSTKÓW =====
            if 13 in lm and 15 in lm and 19 in lm:  # 19 = pinky finger
                left_wrist_angles.append(self._calculate_angle(lm[13], lm[15], lm[19]))
            else:
                left_wrist_angles.append(np.nan)
            
            if 14 in lm and 16 in lm and 20 in lm:
                right_wrist_angles.append(self._calculate_angle(lm[14], lm[16], lm[20]))
            else:
                right_wrist_angles.append(np.nan)
            
            # ===== KĄT TUŁOWIA =====
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                shoulder_center_y = (lm[11]['y'] + lm[12]['y']) / 2
                hip_center_y = (lm[23]['y'] + lm[24]['y']) / 2
                shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
                hip_center_x = (lm[23]['x'] + lm[24]['x']) / 2
                
                torso_angle = np.degrees(np.arctan2(
                    hip_center_y - shoulder_center_y,
                    hip_center_x - shoulder_center_x
                ))
                torso_angles.append(abs(torso_angle))
            else:
                torso_angles.append(np.nan)
            
            # ===== KĄT CIAŁA (ramiona-biodra-kostki) =====
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm and 27 in lm and 28 in lm:
                shoulder_avg = {'x': (lm[11]['x'] + lm[12]['x'])/2, 'y': (lm[11]['y'] + lm[12]['y'])/2}
                hip_avg = {'x': (lm[23]['x'] + lm[24]['x'])/2, 'y': (lm[23]['y'] + lm[24]['y'])/2}
                ankle_avg = {'x': (lm[27]['x'] + lm[28]['x'])/2, 'y': (lm[27]['y'] + lm[28]['y'])/2}
                
                body_angles.append(self._calculate_angle(ankle_avg, hip_avg, shoulder_avg))
            else:
                body_angles.append(np.nan)
            
            # ===== KĄT "DESKI" (nos-biodra-kostki) =====
            if 0 in lm and 23 in lm and 24 in lm and 27 in lm and 28 in lm:
                nose_pt = {'x': lm[0]['x'], 'y': lm[0]['y']}
                hip_avg = {'x': (lm[23]['x'] + lm[24]['x'])/2, 'y': (lm[23]['y'] + lm[24]['y'])/2}
                ankle_avg = {'x': (lm[27]['x'] + lm[28]['x'])/2, 'y': (lm[27]['y'] + lm[28]['y'])/2}
                
                plank_angles.append(self._calculate_angle(ankle_avg, hip_avg, nose_pt))
            else:
                plank_angles.append(np.nan)
            
            # ===== DYSTANSE 3D =====
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                shoulder_center = np.array([
                    (lm[11]['x'] + lm[12]['x']) / 2,
                    (lm[11]['y'] + lm[12]['y']) / 2,
                    (lm[11]['z'] + lm[12]['z']) / 2
                ])
                hip_center = np.array([
                    (lm[23]['x'] + lm[24]['x']) / 2,
                    (lm[23]['y'] + lm[24]['y']) / 2,
                    (lm[23]['z'] + lm[24]['z']) / 2
                ])
                shoulder_hip_distances.append(np.linalg.norm(shoulder_center - hip_center))
            else:
                shoulder_hip_distances.append(np.nan)
            
            if 13 in lm and 14 in lm and 25 in lm and 26 in lm:
                elbow_center = np.array([
                    (lm[13]['x'] + lm[14]['x']) / 2,
                    (lm[13]['y'] + lm[14]['y']) / 2,
                    (lm[13]['z'] + lm[14]['z']) / 2
                ])
                knee_center = np.array([
                    (lm[25]['x'] + lm[26]['x']) / 2,
                    (lm[25]['y'] + lm[26]['y']) / 2,
                    (lm[25]['z'] + lm[26]['z']) / 2
                ])
                elbow_knee_distances.append(np.linalg.norm(elbow_center - knee_center))
            else:
                elbow_knee_distances.append(np.nan)
            
            if 15 in lm and 16 in lm and 11 in lm and 12 in lm:
                wrist_center = np.array([
                    (lm[15]['x'] + lm[16]['x']) / 2,
                    (lm[15]['y'] + lm[16]['y']) / 2,
                    (lm[15]['z'] + lm[16]['z']) / 2
                ])
                shoulder_center = np.array([
                    (lm[11]['x'] + lm[12]['x']) / 2,
                    (lm[11]['y'] + lm[12]['y']) / 2,
                    (lm[11]['z'] + lm[12]['z']) / 2
                ])
                wrist_shoulder_distances.append(np.linalg.norm(wrist_center - shoulder_center))
            else:
                wrist_shoulder_distances.append(np.nan)
            
            # Wysokość klatki (szacunkowa - punkt między ramionami)
            if 11 in lm and 12 in lm:
                chest_y = (lm[11]['y'] + lm[12]['y']) / 2
                chest_ground_distances.append(1.0 - chest_y)  # Odległość od "podłoża" (Y=1)
            else:
                chest_ground_distances.append(np.nan)
        
        # ========== INTERPOLACJA I ZAPISANIE ==========
        # Wysokości Y
        signals['avg_hip_y'] = self._interpolate_nans(np.array(avg_hip_y))
        signals['avg_shoulder_y'] = self._interpolate_nans(np.array(avg_shoulder_y))
        signals['avg_wrist_y'] = self._interpolate_nans(np.array(avg_wrist_y))
        signals['avg_elbow_y'] = self._interpolate_nans(np.array(avg_elbow_y))
        signals['avg_knee_y'] = self._interpolate_nans(np.array(avg_knee_y))
        signals['nose_y'] = self._interpolate_nans(np.array(nose_y))
        
        # Pozycje X i Z
        signals['avg_hip_x'] = self._interpolate_nans(np.array(avg_hip_x))
        signals['avg_shoulder_x'] = self._interpolate_nans(np.array(avg_shoulder_x))
        signals['torso_center_x'] = self._interpolate_nans(np.array(torso_center_x))
        signals['avg_hip_z'] = self._interpolate_nans(np.array(avg_hip_z))
        signals['avg_shoulder_z'] = self._interpolate_nans(np.array(avg_shoulder_z))
        signals['torso_center_z'] = self._interpolate_nans(np.array(torso_center_z))
        
        # Kąty
        signals['left_elbow_angle'] = self._interpolate_nans(np.array(left_elbow_angles))
        signals['right_elbow_angle'] = self._interpolate_nans(np.array(right_elbow_angles))
        signals['avg_elbow_angle'] = (signals['left_elbow_angle'] + signals['right_elbow_angle']) / 2
        signals['left_shoulder_angle'] = self._interpolate_nans(np.array(left_shoulder_angles))
        signals['right_shoulder_angle'] = self._interpolate_nans(np.array(right_shoulder_angles))
        signals['avg_shoulder_angle'] = (signals['left_shoulder_angle'] + signals['right_shoulder_angle']) / 2
        signals['left_wrist_angle'] = self._interpolate_nans(np.array(left_wrist_angles))
        signals['right_wrist_angle'] = self._interpolate_nans(np.array(right_wrist_angles))
        signals['torso_angle'] = self._interpolate_nans(np.array(torso_angles))
        signals['body_angle'] = self._interpolate_nans(np.array(body_angles))
        signals['plank_angle'] = self._interpolate_nans(np.array(plank_angles))
        
        # Dystanse 3D
        signals['shoulder_hip_distance'] = self._interpolate_nans(np.array(shoulder_hip_distances))
        signals['elbow_knee_distance'] = self._interpolate_nans(np.array(elbow_knee_distances))
        signals['wrist_shoulder_distance'] = self._interpolate_nans(np.array(wrist_shoulder_distances))
        signals['chest_ground_distance'] = self._interpolate_nans(np.array(chest_ground_distances))
        
        # Szerokości
        signals['shoulder_width'] = self._interpolate_nans(np.array(shoulder_widths))
        signals['hip_width'] = self._interpolate_nans(np.array(hip_widths))
        signals['elbow_width'] = self._interpolate_nans(np.array(elbow_widths))
        
        # Symetria
        signals['left_right_shoulder_diff'] = self._interpolate_nans(np.array(left_right_shoulder_diff))
        signals['left_right_hip_diff'] = self._interpolate_nans(np.array(left_right_hip_diff))
        
        # Prędkości
        signals['hip_velocity'] = np.gradient(signals['avg_hip_y']) * self.fps
        signals['shoulder_velocity'] = np.gradient(signals['avg_shoulder_y']) * self.fps
        signals['elbow_velocity'] = np.gradient(signals['avg_elbow_y']) * self.fps
        signals['chest_velocity'] = np.gradient(signals['chest_ground_distance']) * self.fps
        
        # Przyspieszenia (drugie pochodne)
        signals['hip_acceleration'] = np.gradient(signals['hip_velocity']) * self.fps
        
        # Synchronizacja
        signals['elbow_sync'] = np.abs(signals['left_elbow_angle'] - signals['right_elbow_angle'])
        signals['shoulder_sync'] = np.abs(signals['left_shoulder_angle'] - signals['right_shoulder_angle'])
        
        # Zapisz orientację dla późniejszego użycia
        signals['camera_orientation'] = camera_orientation
        
        return signals
    
    def _detect_pushup_repetitions(self, signals: Dict[str, np.ndarray],
                                   landmarks_data: List[Dict]) -> List[Dict]:
        """
        Stabilna detekcja powtórzeń pompek.
        Wzorzec: PEAK → VALLEY → PEAK (góra → dół → góra).
        """

        fps = self.fps

        # 1. Wybór sygnału głównego – ten z największą amplitudą
        base_signal_candidates = [
            'avg_hip_y',
            'avg_shoulder_y',
            'chest_ground_distance',
            'avg_elbow_angle'
        ]

        available = [s for s in base_signal_candidates if s in signals]
        if not available:
            return []

        amplitudes = {s: np.ptp(signals[s]) for s in available}
        primary_signal = max(amplitudes, key=amplitudes.get)

        raw = np.array(signals[primary_signal])
        if np.ptp(raw) < 1e-4:
            return []

        # dla sygnałów Y odwracamy oś (małe Y = góra)
        if primary_signal in ['avg_hip_y', 'avg_shoulder_y']:
            raw = -raw

        # 2. Wygładzenie
        sigma = fps * 0.08
        smoothed = gaussian_filter1d(raw, sigma=sigma)

        # 3. Detekcja peaków i dolin
        min_distance = int(fps * 0.35)

        peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=0.04)
        valleys, _ = find_peaks(-smoothed, distance=min_distance, prominence=0.04)

        if len(peaks) == 0 or len(valleys) == 0:
            return []

        events = [('peak', int(p)) for p in peaks] + \
                 [('valley', int(v)) for v in valleys]
        events.sort(key=lambda x: x[1])

        repetitions = []
        rep_id = 1

        i = 0
        while i < len(events) - 2:
            e1, e2, e3 = events[i], events[i+1], events[i+2]

            if e1[0] == 'peak' and e2[0] == 'valley' and e3[0] == 'peak':
                start = e1[1]
                mid = e2[1]
                end = e3[1]

                dur = (end - start) / fps
                if not (self.min_rep_duration <= dur <= self.max_rep_duration):
                    i += 1
                    continue

                up = smoothed[start]
                down = smoothed[mid]
                amp = abs(up - down)

                # amplituda musi być sensowna
                if amp < 0.10 * np.ptp(smoothed):
                    i += 1
                    continue

                # 4. Konsensus z innych sygnałów
                if not self._consensus_support(signals, start, end):
                    i += 1
                    continue

                repetitions.append({
                    'rep_id': rep_id,
                    'start_frame': int(start),
                    'end_frame': int(end),
                    'start_time': start / fps,
                    'end_time': end / fps,
                    'duration': dur,
                    'signal_used': primary_signal,
                    'consensus_score': 0.0  # w razie czego możesz później podbić
                })

                rep_id += 1
                i += 2
            else:
                i += 1

        return repetitions
    
    def _validate_pushup_repetitions(self, repetitions: List[Dict], landmarks_data: List[Dict]) -> List[Dict]:
        """Walidacja jakości powtórzeń pompek - Z DEDYKOWANEGO PROCESORA"""
        validated = []
        
        for rep in repetitions:
            start_frame = rep['start_frame']
            end_frame = rep['end_frame']
            
            rep_landmarks = landmarks_data[start_frame:end_frame+1]
            quality_score = self._calculate_quality_score(rep_landmarks)
            
            rep['quality_score'] = quality_score
            
            if quality_score >= 0.3:
                validated.append(rep)
        
        return validated
    
    def _calculate_quality_score(self, rep_landmarks: List[Dict]) -> float:
        """Oblicz score jakości - Z DEDYKOWANEGO PROCESORA"""
        if len(rep_landmarks) < 3:
            return 0.0
        
        scores = []
        
        # Symetryczność
        left_angles = []
        right_angles = []
        
        for data in rep_landmarks:
            lm = data['landmarks']
            if 11 in lm and 13 in lm and 15 in lm:
                left_angles.append(self._calculate_angle(lm[11], lm[13], lm[15]))
            if 12 in lm and 14 in lm and 16 in lm:
                right_angles.append(self._calculate_angle(lm[12], lm[14], lm[16]))
        
        if len(left_angles) > 0 and len(right_angles) > 0:
            symmetry = 1.0 - (np.std(np.array(left_angles) - np.array(right_angles)) / 180.0)
            scores.append(max(0, symmetry))
        
        # Zakres ruchu
        hip_heights = []
        for data in rep_landmarks:
            lm = data['landmarks']
            if 23 in lm and 24 in lm:
                hip_heights.append((lm[23]['y'] + lm[24]['y']) / 2)
        
        if len(hip_heights) > 0:
            rom = np.max(hip_heights) - np.min(hip_heights)
            rom_score = min(1.0, rom / 0.15)
            scores.append(rom_score)
        
        # Płynność
        if len(hip_heights) > 2:
            velocity = np.diff(hip_heights)
            acceleration = np.diff(velocity)
            smoothness = 1.0 - min(1.0, np.std(acceleration) / 0.05)
            scores.append(max(0, smoothness))
        
        return np.mean(scores) if scores else 0.0
    def _consensus_support(self, signals: Dict[str, np.ndarray], start: int, end: int) -> bool:
        """
        Sygnały muszą zgodzić się co do istnienia ruchu góra-dół-góra.
        Wystarczą 2 sygnały poza głównym.
        """

        support = 0
        window = slice(start, end+1)

        backup_signals = [
            'avg_shoulder_y', 'avg_hip_y',
            'chest_ground_distance', 'avg_elbow_angle',
            'avg_elbow_y', 'torso_angle'
        ]

        for s in backup_signals:
            if s not in signals:
                continue

            sig = np.array(signals[s])[window]
            if np.ptp(sig) < 1e-4:
                continue

            # normalizacja
            r = sig - np.min(sig)
            r = r / (np.max(sig) - np.min(sig) + 1e-6)

            # muszą być 2 ekstrema (min i max)
            if abs(np.max(r) - np.min(r)) < 0.25:
                continue

            support += 1

        return support >= 2

    
    def _calculate_pushup_metrics(self, rep_landmarks: List[Dict]) -> Dict:
        """Oblicz szczegółowe metryki dla CSV - Z DEDYKOWANEGO PROCESORA"""
        metrics = {
            'primary_joint_range': 0.0,
            'primary_angle_mean': 0.0,
            'primary_angle_std': 0.0,
            'secondary_angle_mean': 0.0,
            'symmetry_score': 0.0
        }
        
        if not rep_landmarks:
            return metrics
        
        # Hip range
        hip_heights = []
        elbow_angles = []
        torso_angles = []
        left_elbow = []
        right_elbow = []
        
        for data in rep_landmarks:
            lm = data['landmarks']
            
            if 23 in lm and 24 in lm:
                hip_heights.append((lm[23]['y'] + lm[24]['y']) / 2)
            
            if 11 in lm and 13 in lm and 15 in lm:
                angle = self._calculate_angle(lm[11], lm[13], lm[15])
                left_elbow.append(angle)
                elbow_angles.append(angle)
            
            if 12 in lm and 14 in lm and 16 in lm:
                angle = self._calculate_angle(lm[12], lm[14], lm[16])
                right_elbow.append(angle)
                elbow_angles.append(angle)
            
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                shoulder_center_y = (lm[11]['y'] + lm[12]['y']) / 2
                hip_center_y = (lm[23]['y'] + lm[24]['y']) / 2
                shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
                hip_center_x = (lm[23]['x'] + lm[24]['x']) / 2
                
                angle = np.degrees(np.arctan2(
                    hip_center_y - shoulder_center_y,
                    hip_center_x - shoulder_center_x
                ))
                torso_angles.append(abs(angle))
        
        if hip_heights:
            metrics['primary_joint_range'] = np.max(hip_heights) - np.min(hip_heights)
        
        if elbow_angles:
            metrics['primary_angle_mean'] = np.mean(elbow_angles)
            metrics['primary_angle_std'] = np.std(elbow_angles)
        
        if torso_angles:
            metrics['secondary_angle_mean'] = np.mean(torso_angles)
        
        if left_elbow and right_elbow:
            diff = np.abs(np.array(left_elbow) - np.array(right_elbow))
            metrics['symmetry_score'] = 1.0 - (np.mean(diff) / 180.0)
        
        return metrics
    
    # ==================== PRZYSIADY ==================== 
    
    def _compute_squat_signals(self, landmarks_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Oblicz sygnały ruchu dla przysiadów"""
        signals = {}
        
        avg_hip_y = []
        avg_knee_y = []
        left_knee_angles = []
        right_knee_angles = []
        left_hip_angles = []
        right_hip_angles = []
        torso_angles = []
        squat_depth_ratios = []
        hip_knee_distances = []
        
        # Pięty
        left_heel_y = []
        right_heel_y = []
        avg_heel_y = []
        
        for data in landmarks_data:
            lm = data['landmarks']
            
            # Biodra
            if 23 in lm and 24 in lm:
                avg_hip_y.append((lm[23]['y'] + lm[24]['y']) / 2)
            else:
                avg_hip_y.append(np.nan)
            
            # Kolana
            if 25 in lm and 26 in lm:
                avg_knee_y.append((lm[25]['y'] + lm[26]['y']) / 2)
            else:
                avg_knee_y.append(np.nan)
            
            # Dystans 3D biodro-kolano
            if 23 in lm and 24 in lm and 25 in lm and 26 in lm:
                hip_center = np.array([
                    (lm[23]['x'] + lm[24]['x']) / 2,
                    (lm[23]['y'] + lm[24]['y']) / 2,
                    (lm[23]['z'] + lm[24]['z']) / 2
                ])
                knee_center = np.array([
                    (lm[25]['x'] + lm[26]['x']) / 2,
                    (lm[25]['y'] + lm[26]['y']) / 2,
                    (lm[25]['z'] + lm[26]['z']) / 2
                ])
                distance = np.linalg.norm(hip_center - knee_center)
                hip_knee_distances.append(distance)
            else:
                hip_knee_distances.append(np.nan)
            
            # Pięty
            if 29 in lm:
                left_heel_y.append(lm[29]['y'])
            else:
                left_heel_y.append(np.nan)
                
            if 30 in lm:
                right_heel_y.append(lm[30]['y'])
            else:
                right_heel_y.append(np.nan)
            
            if 29 in lm and 30 in lm:
                avg_heel_y.append((lm[29]['y'] + lm[30]['y']) / 2)
            else:
                avg_heel_y.append(np.nan)
            
            # Kąty kolan
            if 23 in lm and 25 in lm and 27 in lm:
                left_knee_angles.append(self._calculate_angle(lm[23], lm[25], lm[27]))
            else:
                left_knee_angles.append(np.nan)
            
            if 24 in lm and 26 in lm and 28 in lm:
                right_knee_angles.append(self._calculate_angle(lm[24], lm[26], lm[28]))
            else:
                right_knee_angles.append(np.nan)
            
            # Kąty bioder
            if 11 in lm and 23 in lm and 25 in lm:
                left_hip_angles.append(self._calculate_angle(lm[11], lm[23], lm[25]))
            else:
                left_hip_angles.append(np.nan)
            
            if 12 in lm and 24 in lm and 26 in lm:
                right_hip_angles.append(self._calculate_angle(lm[12], lm[24], lm[26]))
            else:
                right_hip_angles.append(np.nan)
            
            # Kąt tułowia
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                shoulder_center_y = (lm[11]['y'] + lm[12]['y']) / 2
                hip_center_y = (lm[23]['y'] + lm[24]['y']) / 2
                shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
                hip_center_x = (lm[23]['x'] + lm[24]['x']) / 2
                
                torso_angle = np.degrees(np.arctan2(
                    hip_center_y - shoulder_center_y,
                    hip_center_x - shoulder_center_x
                ))
                torso_angles.append(abs(90 - abs(torso_angle)))
            else:
                torso_angles.append(np.nan)
            
            # Głębokość
            if 23 in lm and 24 in lm and 25 in lm and 26 in lm:
                avg_hip = (lm[23]['y'] + lm[24]['y']) / 2
                avg_knee = (lm[25]['y'] + lm[26]['y']) / 2
                if avg_knee > 0:
                    squat_depth_ratios.append(avg_hip / avg_knee)
                else:
                    squat_depth_ratios.append(np.nan)
            else:
                squat_depth_ratios.append(np.nan)
        
        # Interpolacja
        signals['avg_hip_y'] = self._interpolate_nans(np.array(avg_hip_y))
        signals['avg_knee_y'] = self._interpolate_nans(np.array(avg_knee_y))
        signals['hip_knee_distance'] = self._interpolate_nans(np.array(hip_knee_distances))
        signals['left_knee_angle'] = self._interpolate_nans(np.array(left_knee_angles))
        signals['right_knee_angle'] = self._interpolate_nans(np.array(right_knee_angles))
        signals['avg_knee_angle'] = (signals['left_knee_angle'] + signals['right_knee_angle']) / 2
        signals['left_hip_angle'] = self._interpolate_nans(np.array(left_hip_angles))
        signals['right_hip_angle'] = self._interpolate_nans(np.array(right_hip_angles))
        signals['avg_hip_angle'] = (signals['left_hip_angle'] + signals['right_hip_angle']) / 2
        signals['torso_angle'] = self._interpolate_nans(np.array(torso_angles))
        signals['squat_depth_ratio'] = self._interpolate_nans(np.array(squat_depth_ratios))
        
        # Prędkości
        signals['hip_velocity'] = np.gradient(signals['avg_hip_y']) * self.fps
        signals['knee_sync'] = np.abs(signals['left_knee_angle'] - signals['right_knee_angle'])
        
        # Sygnały pięt
        signals['left_heel_y'] = self._interpolate_nans(np.array(left_heel_y))
        signals['right_heel_y'] = self._interpolate_nans(np.array(right_heel_y))
        signals['avg_heel_y'] = self._interpolate_nans(np.array(avg_heel_y))
        signals['left_heel_lift'] = -np.gradient(signals['left_heel_y']) * self.fps
        signals['right_heel_lift'] = -np.gradient(signals['right_heel_y']) * self.fps
        signals['avg_heel_lift'] = (signals['left_heel_lift'] + signals['right_heel_lift']) / 2
        
        if len(signals['avg_heel_y']) > 0:
            baseline_heel_y = np.percentile(signals['avg_heel_y'], 90)
            signals['heel_elevation'] = baseline_heel_y - signals['avg_heel_y']
        else:
            signals['heel_elevation'] = np.zeros_like(signals['avg_heel_y'])
        
        return signals
    
    def _detect_squat_repetitions(self, signals: Dict[str, np.ndarray], landmarks_data: List[Dict]) -> List[Dict]:
        """Detekcja powtórzeń przysiadów - ADAPTACYJNA WERSJA"""
        
        # Określ priorytet na podstawie analizy sygnałów
        signal_amplitudes = {}
        for signal_name in ['avg_knee_angle', 'avg_hip_angle', 'hip_knee_distance', 'avg_hip_y']:
            if signal_name in signals:
                signal = signals[signal_name]
                amplitude = np.max(signal) - np.min(signal)
                signal_amplitudes[signal_name] = amplitude
        
        if self.debug:
            print("\n📊 Amplitudy sygnałów:")
            for name, amp in sorted(signal_amplitudes.items(), key=lambda x: x[1], reverse=True):
                print(f"   {name}: {amp:.4f}")
        
        # Sortuj sygnały według amplitudy
        sorted_signals = sorted(signal_amplitudes.items(), key=lambda x: x[1], reverse=True)
        
        # Dynamiczny priorytet
        signal_priority = []
        for idx, (signal_name, amplitude) in enumerate(sorted_signals):
            if signal_name == 'avg_knee_angle':
                weight = 1.0 - (idx * 0.1)
                polarity = 'inverted'
            elif signal_name == 'avg_hip_angle':
                weight = 0.9 - (idx * 0.1)
                polarity = 'inverted'
            elif signal_name == 'hip_knee_distance':
                weight = 0.85 - (idx * 0.1)
                polarity = 'inverted'
            elif signal_name == 'avg_hip_y':
                weight = 0.6 - (idx * 0.1)
                polarity = 'normal'
            else:
                continue
            
            signal_priority.append((signal_name, max(0.3, weight), polarity))
        
        # Dodaj depth ratio
        if 'squat_depth_ratio' in signals:
            signal_priority.append(('squat_depth_ratio', 0.3, 'normal'))
        
        if self.debug:
            print("\n🎯 Dynamiczny priorytet sygnałów:")
            for name, weight, polarity in signal_priority:
                print(f"   {name}: weight={weight:.2f}, polarity={polarity}")
        
        # Detekcja z prostszą logiką (bez peak-valley-peak)
        all_detections = {}
        
        for signal_name, weight, polarity in signal_priority:
            if signal_name not in signals:
                continue
            
            signal = signals[signal_name]
            if len(signal) == 0 or np.all(np.isnan(signal)):
                continue
            
            signal_range = np.max(signal) - np.min(signal)
            if signal_range < 1e-6:
                continue
            
            normalized = (signal - np.min(signal)) / signal_range
            
            if polarity == 'inverted':
                normalized = 1.0 - normalized
            
            smoothed = gaussian_filter1d(normalized, sigma=self.fps * 0.12)
            
            peaks, _ = find_peaks(smoothed, distance=int(self.fps * 0.4), 
                                 prominence=self.peak_prominence, width=3)
            valleys, _ = find_peaks(-smoothed, distance=int(self.fps * 0.4), 
                                   prominence=self.peak_prominence, width=3)
            
            all_detections[signal_name] = {
                'peaks': peaks,
                'valleys': valleys,
                'smoothed': smoothed,
                'weight': weight,
                'polarity': polarity
            }
        
        if not all_detections:
            return []
        
        # Dla przysiadów używamy valley-peak-valley
        repetitions = self._find_squat_consensus_repetitions(all_detections)
        
        return repetitions
    
    def _find_squat_consensus_repetitions(self, all_detections: Dict) -> List[Dict]:
        """Znajdź powtórzenia przysiadów - wzorzec valley-peak-valley"""
        
        primary_signal = list(all_detections.keys())[0]
        primary = all_detections[primary_signal]
        
        peaks = primary['peaks']
        valleys = primary['valleys']
        
        events = []
        for peak in peaks:
            events.append(('peak', peak))
        for valley in valleys:
            events.append(('valley', valley))
        
        events.sort(key=lambda x: x[1])
        
        repetitions = []
        rep_id = 1
        
        # Początek wideo
        if len(events) > 0 and events[0][0] == 'peak':
            if len(events) > 1 and events[1][0] == 'valley':
                next_valley = events[1][1]
                duration = next_valley / self.fps
                
                if self.min_rep_duration <= duration <= self.max_rep_duration:
                    if self._validate_by_consensus(0, next_valley, all_detections):
                        repetitions.append({
                            'rep_id': rep_id,
                            'start_frame': 0,
                            'end_frame': int(next_valley),
                            'start_time': 0.0,
                            'end_time': next_valley / self.fps,
                            'duration': duration,
                            'signal_used': primary_signal,
                            'note': 'edge_start',
                            'consensus_score': self._calculate_consensus_score(0, next_valley, all_detections)
                        })
                        rep_id += 1
        
        # Normalne powtórzenia (valley-peak-valley)
        i = 0
        while i < len(events) - 2:
            if (events[i][0] == 'valley' and events[i+1][0] == 'peak' and events[i+2][0] == 'valley'):
                start_frame = events[i][1]
                end_frame = events[i+2][1]
                duration = (end_frame - start_frame) / self.fps
                
                if self.min_rep_duration <= duration <= self.max_rep_duration:
                    if self._validate_by_consensus(start_frame, end_frame, all_detections):
                        repetitions.append({
                            'rep_id': rep_id,
                            'start_frame': int(start_frame),
                            'end_frame': int(end_frame),
                            'start_time': start_frame / self.fps,
                            'end_time': end_frame / self.fps,
                            'duration': duration,
                            'signal_used': primary_signal,
                            'consensus_score': self._calculate_consensus_score(start_frame, end_frame, all_detections)
                        })
                        rep_id += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        # Koniec wideo
        if len(events) > 0 and events[-1][0] == 'peak':
            last_frame = len(primary['smoothed']) - 1
            if len(events) > 1 and events[-2][0] == 'valley':
                prev_valley = events[-2][1]
                duration = (last_frame - prev_valley) / self.fps
                
                if self.min_rep_duration <= duration <= self.max_rep_duration:
                    if self._validate_by_consensus(prev_valley, last_frame, all_detections):
                        repetitions.append({
                            'rep_id': rep_id,
                            'start_frame': int(prev_valley),
                            'end_frame': int(last_frame),
                            'start_time': prev_valley / self.fps,
                            'end_time': last_frame / self.fps,
                            'duration': duration,
                            'signal_used': primary_signal,
                            'note': 'edge_end',
                            'consensus_score': self._calculate_consensus_score(prev_valley, last_frame, all_detections)
                        })
        
        return repetitions
    
    def _calculate_consensus_score(self, start_frame: int, end_frame: int, all_detections: Dict) -> float:
        """Oblicz score konsensusu - Z DEDYKOWANEGO PROCESORA"""
        votes = 0
        total_weight = 0
        
        for signal_name, detection in all_detections.items():
            weight = detection['weight']
            total_weight += weight
            
            peaks_in_range = detection['peaks'][
                (detection['peaks'] >= start_frame) & (detection['peaks'] <= end_frame)
            ]
            valleys_in_range = detection['valleys'][
                (detection['valleys'] >= start_frame) & (detection['valleys'] <= end_frame)
            ]
            
            if len(peaks_in_range) > 0 or len(valleys_in_range) > 0:
                votes += weight
        
        return votes / total_weight if total_weight > 0 else 0.0
    
    def _validate_squat_repetitions(self, repetitions: List[Dict], landmarks_data: List[Dict]) -> List[Dict]:
        """Walidacja jakości powtórzeń przysiadów"""
        validated = []
        
        for rep in repetitions:
            start_frame = rep['start_frame']
            end_frame = rep['end_frame']
            rep_landmarks = landmarks_data[start_frame:end_frame+1]
            
            quality_metrics = self._calculate_squat_quality(rep_landmarks)
            rep.update(quality_metrics)
            
            if quality_metrics['quality_score'] >= 0.25:
                validated.append(rep)
        
        return validated
    
    def _calculate_squat_quality(self, rep_landmarks: List[Dict]) -> Dict:
        """Oblicz metryki jakości przysiadu"""
        if len(rep_landmarks) < 3:
            return {
                'quality_score': 0.0,
                'depth_category': 'unknown',
                'min_knee_angle': None,
                'max_heel_lift': 0.0
            }
        
        scores = []
        
        # Symetryczność kolan
        left_knee_angles = []
        right_knee_angles = []
        left_heel_heights = []
        right_heel_heights = []
        
        for data in rep_landmarks:
            lm = data['landmarks']
            
            if 23 in lm and 25 in lm and 27 in lm:
                left_knee_angles.append(self._calculate_angle(lm[23], lm[25], lm[27]))
            
            if 24 in lm and 26 in lm and 28 in lm:
                right_knee_angles.append(self._calculate_angle(lm[24], lm[26], lm[28]))
            
            if 29 in lm:
                left_heel_heights.append(lm[29]['y'])
            
            if 30 in lm:
                right_heel_heights.append(lm[30]['y'])
        
        symmetry_score = 0.0
        if len(left_knee_angles) > 0 and len(right_knee_angles) > 0:
            symmetry = 1.0 - (np.std(np.array(left_knee_angles) - np.array(right_knee_angles)) / 180.0)
            symmetry_score = max(0, symmetry)
            scores.append(symmetry_score)
        
        # Głębokość
        min_knee_angle = None
        depth_category = 'unknown'
        
        if left_knee_angles and right_knee_angles:
            min_knee_angle = min(min(left_knee_angles), min(right_knee_angles))
            
            if min_knee_angle < 90:
                depth_score = 1.0
                depth_category = 'deep'
            elif min_knee_angle < 110:
                depth_score = 0.8
                depth_category = 'parallel'
            else:
                depth_score = 0.5
                depth_category = 'shallow'
            
            scores.append(depth_score)
        
        # Zakres ruchu bioder
        hip_heights = []
        for data in rep_landmarks:
            lm = data['landmarks']
            if 23 in lm and 24 in lm:
                hip_heights.append((lm[23]['y'] + lm[24]['y']) / 2)
        
        if len(hip_heights) > 0:
            rom = np.max(hip_heights) - np.min(hip_heights)
            rom_score = min(1.0, rom / 0.20)
            scores.append(rom_score)
        
        # Płynność
        if len(hip_heights) > 2:
            velocity = np.diff(hip_heights)
            acceleration = np.diff(velocity)
            smoothness = 1.0 - min(1.0, np.std(acceleration) / 0.08)
            scores.append(max(0, smoothness))
        
        # Pięty
        max_heel_lift = 0.0
        if left_heel_heights and right_heel_heights:
            left_baseline = np.max(left_heel_heights)
            right_baseline = np.max(right_heel_heights)
            left_max_lift = left_baseline - np.min(left_heel_heights)
            right_max_lift = right_baseline - np.min(right_heel_heights)
            max_heel_lift = max(left_max_lift, right_max_lift)
            
            if max_heel_lift > 0.05:
                heel_penalty = max(0, 1.0 - (max_heel_lift - 0.05) / 0.15)
                scores.append(heel_penalty)
        
        quality_score = np.mean(scores) if scores else 0.0
        
        return {
            'quality_score': quality_score,
            'symmetry_score': symmetry_score,
            'depth_category': depth_category,
            'min_knee_angle': min_knee_angle,
            'max_heel_lift': max_heel_lift
        }
    
    def _calculate_squat_metrics(self, rep_landmarks: List[Dict]) -> Dict:
        """Oblicz szczegółowe metryki dla przysiadów"""
        metrics = {
            'primary_joint_range': 0.0,
            'primary_angle_mean': 0.0,
            'primary_angle_std': 0.0,
            'secondary_angle_mean': 0.0,
            'symmetry_score': 0.0
        }
        
        if not rep_landmarks:
            return metrics
        
        hip_heights = []
        knee_angles = []
        torso_angles = []
        left_knee = []
        right_knee = []
        
        for data in rep_landmarks:
            lm = data['landmarks']
            
            if 23 in lm and 24 in lm:
                hip_heights.append((lm[23]['y'] + lm[24]['y']) / 2)
            
            if 23 in lm and 25 in lm and 27 in lm:
                angle = self._calculate_angle(lm[23], lm[25], lm[27])
                left_knee.append(angle)
                knee_angles.append(angle)
            
            if 24 in lm and 26 in lm and 28 in lm:
                angle = self._calculate_angle(lm[24], lm[26], lm[28])
                right_knee.append(angle)
                knee_angles.append(angle)
            
            if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
                shoulder_center_y = (lm[11]['y'] + lm[12]['y']) / 2
                hip_center_y = (lm[23]['y'] + lm[24]['y']) / 2
                shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
                hip_center_x = (lm[23]['x'] + lm[24]['x']) / 2
                
                angle = np.degrees(np.arctan2(
                    hip_center_y - shoulder_center_y,
                    hip_center_x - shoulder_center_x
                ))
                torso_angles.append(abs(90 - abs(angle)))
        
        if hip_heights:
            metrics['primary_joint_range'] = np.max(hip_heights) - np.min(hip_heights)
        
        if knee_angles:
            metrics['primary_angle_mean'] = np.mean(knee_angles)
            metrics['primary_angle_std'] = np.std(knee_angles)
        
        if torso_angles:
            metrics['secondary_angle_mean'] = np.mean(torso_angles)
        
        if left_knee and right_knee:
            diff = np.abs(np.array(left_knee) - np.array(right_knee))
            metrics['symmetry_score'] = 1.0 - (np.mean(diff) / 180.0)
        
        return metrics
    
    # ==================== WSPÓLNE METODY ==================== 
    
    def _extract_video_clip(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        output_path: Path,
        width: int,
        height: int,
        fps: float
    ):
        """
        Wytnij klip wideo.

        ✅ Jeśli jest FFmpeg:
            - wycina po czasie (ss/to)
            - POZWALA FFmpegOWI obsłużyć rotację tak, jak widzi ją normalny odtwarzacz
        ✅ Jeśli nie ma FFmpeg:
            - fallback: zapis klatek przez OpenCV (może różnić się orientacją na iPhone)
        """

        # GŁÓWNA ŚCIEŻKA – FFmpeg, żeby klip wyglądał jak w odtwarzaczu
        if self.has_ffmpeg and hasattr(self, "_current_video_path"):
            input_path = self._current_video_path

            start_time = max(0.0, start_frame / fps)
            end_time = max(start_time, end_frame / fps)

            self._ffmpeg_extract_segment(
                input_path=input_path,
                output_path=str(output_path),
                start_time=start_time,
                end_time=end_time
            )
            return

        # Fallback – stary sposób przez OpenCV
        temp_file = output_path.with_suffix('.avi')

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(temp_file), fourcc, fps, (width, height))

        if not out.isOpened():
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

        if self.has_ffmpeg:
            self._convert_to_h264(str(temp_file), str(output_path))
        else:
            temp_file.rename(output_path.with_suffix('.avi'))
    def _extract_video_clip_with_visualization(
        self,
        video_path: str,
        landmarks_data: List[Dict],
        start_frame: int,
        end_frame: int,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        rep_info: Dict
    ):
        """
        Wytnij klip wideo Z NAŁOŻONYMI PUNKTAMI MediaPipe i metrykami
        
        Args:
            video_path: Ścieżka do oryginalnego wideo
            landmarks_data: Lista danych landmarks
            start_frame: Klatka początkowa
            end_frame: Klatka końcowa
            output_path: Ścieżka wyjściowa
            width, height: Wymiary wideo
            fps: Klatki na sekundę
            rep_info: Słownik z informacjami o powtórzeniu
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        rotation = self._detect_video_rotation(video_path)
        
        temp_file = output_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        if rotation in [90, 270]:
            out_width, out_height = height, width
        else:
            out_width, out_height = width, height
        
        out = cv2.VideoWriter(str(temp_file), fourcc, fps, (out_width, out_height))
        
        if not out.isOpened():
            print(f"⚠️  Nie udało się utworzyć VideoWriter")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_offset in range(end_frame - start_frame + 1):
            current_frame_idx = start_frame + frame_offset
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if rotation != 0:
                frame = self._rotate_frame(frame, rotation)
            
            # Znajdź landmarks dla tej klatki
            landmarks_for_frame = None
            for lm_data in landmarks_data:
                if lm_data['frame'] == current_frame_idx:
                    landmarks_for_frame = lm_data['landmarks']
                    break
            
            # Rysuj landmarks i metryki
            if landmarks_for_frame is not None:
                frame = self._draw_landmarks_and_metrics(
                    frame, 
                    landmarks_for_frame, 
                    rep_info,
                    frame_offset,
                    end_frame - start_frame + 1
                )
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        if self.has_ffmpeg:
            self._convert_to_h264(str(temp_file), str(output_path))
        else:
            temp_file.rename(output_path.with_suffix('.avi'))

    def _create_full_video_with_landmarks(
            self,
            video_path: str,
            landmarks_data: List[Dict],
            output_path: Path
    ):
        """
        Stwórz pełne wideo z nałożonymi landmarkami (bez wycinania)

        Args:
            video_path: Ścieżka do oryginalnego wideo
            landmarks_data: Lista danych landmarks
            output_path: Ścieżka wyjściowa
        """
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = self._detect_video_rotation(video_path)

        # Wymiary po rotacji
        if rotation in [90, 270]:
            out_width, out_height = height, width
        else:
            out_width, out_height = width, height

        # Tymczasowy plik AVI
        temp_file = output_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(temp_file), fourcc, fps, (out_width, out_height))

        if not out.isOpened():
            print(f"⚠️  Nie udało się utworzyć VideoWriter dla pełnego wideo")
            cap.release()
            return

        # Stwórz mapę: numer_klatki -> landmarks
        landmarks_map = {lm_data['frame']: lm_data['landmarks'] for lm_data in landmarks_data}

        frame_idx = 0
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Rotacja (jeśli potrzebna)
            if rotation != 0:
                frame = self._rotate_frame(frame, rotation)

            # Jeśli mamy landmarks dla tej klatki, nałóż je
            if frame_idx in landmarks_map:
                landmarks = landmarks_map[frame_idx]
                frame = self._draw_landmarks_simple(frame, landmarks)

            out.write(frame)

            frame_idx += 1
            processed_frames += 1

            # Progress
            if frame_idx % 30 == 0 and self.debug:
                progress = (frame_idx / total_frames) * 100
                print(f"  Postęp wizualizacji: {progress:.1f}%", end='\r')

        cap.release()
        out.release()

        if self.debug:
            print(f"\n  ✅ Przetworzone klatki: {processed_frames}/{total_frames}")

        # Konwersja do H.264
        if self.has_ffmpeg:
            self._convert_to_h264(str(temp_file), str(output_path))
        else:
            temp_file.rename(output_path.with_suffix('.avi'))

    def _draw_landmarks_simple(
            self,
            frame: np.ndarray,
            landmarks: Dict
    ) -> np.ndarray:
        """
        Rysuj tylko punkty MediaPipe (bez metryk i panelu)
        Szybsza wersja dla pełnego wideo
        """
        frame_height, frame_width = frame.shape[:2]

        # Kolory części ciała
        COLORS = {
            'face': (255, 200, 200),
            'torso': (0, 255, 0),
            'arms': (0, 200, 255),
            'legs': (255, 100, 100),
            'feet': (200, 0, 255),
        }

        BODY_PARTS = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'torso': [11, 12, 23, 24],
            'arms': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'legs': [25, 26, 27, 28],
            'feet': [29, 30, 31, 32]
        }

        CONNECTIONS = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        ]

        # Rysuj połączenia
        for idx1, idx2 in CONNECTIONS:
            if idx1 in landmarks and idx2 in landmarks:
                lm1, lm2 = landmarks[idx1], landmarks[idx2]
                if lm1['visibility'] > 0.5 and lm2['visibility'] > 0.5:
                    x1 = int(lm1['x'] * frame_width)
                    y1 = int(lm1['y'] * frame_height)
                    x2 = int(lm2['x'] * frame_width)
                    y2 = int(lm2['y'] * frame_height)
                    cv2.line(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)

        # Rysuj punkty
        for part_name, indices in BODY_PARTS.items():
            color = COLORS[part_name]
            for idx in indices:
                if idx in landmarks and landmarks[idx]['visibility'] > 0.5:
                    x = int(landmarks[idx]['x'] * frame_width)
                    y = int(landmarks[idx]['y'] * frame_height)
                    radius = 5 if idx in [11, 12, 13, 14, 23, 24, 25, 26] else 3
                    cv2.circle(frame, (x, y), radius, color, -1)
                    cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)

        # Dodaj małą notatkę w rogu
        cv2.putText(frame, "MediaPipe Pose", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return frame
    def _draw_landmarks_and_metrics(
        self,
        frame: np.ndarray,
        landmarks: Dict,
        rep_info: Dict,
        current_frame: int,
        total_frames: int
    ) -> np.ndarray:
        """Rysuj punkty MediaPipe i metryki na klatce"""
        import cv2
        
        frame_height, frame_width = frame.shape[:2]
        
        # Kolory części ciała
        COLORS = {
            'face': (255, 200, 200),
            'torso': (0, 255, 0),
            'arms': (0, 200, 255),
            'legs': (255, 100, 100),
            'feet': (200, 0, 255),
        }
        
        BODY_PARTS = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'torso': [11, 12, 23, 24],
            'arms': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'legs': [25, 26, 27, 28],
            'feet': [29, 30, 31, 32]
        }
        
        CONNECTIONS = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        ]
        
        # Rysuj połączenia
        for idx1, idx2 in CONNECTIONS:
            if idx1 in landmarks and idx2 in landmarks:
                lm1, lm2 = landmarks[idx1], landmarks[idx2]
                if lm1['visibility'] > 0.5 and lm2['visibility'] > 0.5:
                    x1 = int(lm1['x'] * frame_width)
                    y1 = int(lm1['y'] * frame_height)
                    x2 = int(lm2['x'] * frame_width)
                    y2 = int(lm2['y'] * frame_height)
                    cv2.line(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
        
        # Rysuj punkty
        for part_name, indices in BODY_PARTS.items():
            color = COLORS[part_name]
            for idx in indices:
                if idx in landmarks and landmarks[idx]['visibility'] > 0.5:
                    x = int(landmarks[idx]['x'] * frame_width)
                    y = int(landmarks[idx]['y'] * frame_height)
                    radius = 6 if idx in [11, 12, 13, 14, 23, 24, 25, 26] else 3
                    cv2.circle(frame, (x, y), radius, color, -1)
        
        # Panel metryk
        progress = (current_frame / total_frames) * 100
        metrics_text = [
            f"Postep: {progress:.0f}%",
            f"Klatka: {current_frame}/{total_frames}",
            f"Rep #{rep_info.get('rep_number', '?')}",
            f"Quality: {rep_info.get('quality_score', 0):.2f}"
        ]
        
        # Dodaj kąty
        if self.exercise_type == 'pushup':
            if 11 in landmarks and 13 in landmarks and 15 in landmarks:
                angle = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                metrics_text.append(f"L Elbow: {angle:.0f}deg")
            if 12 in landmarks and 14 in landmarks and 16 in landmarks:
                angle = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                metrics_text.append(f"R Elbow: {angle:.0f}deg")
        elif self.exercise_type == 'squat':
            if 23 in landmarks and 25 in landmarks and 27 in landmarks:
                angle = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                metrics_text.append(f"L Knee: {angle:.0f}deg")
            if 24 in landmarks and 26 in landmarks and 28 in landmarks:
                angle = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                metrics_text.append(f"R Knee: {angle:.0f}deg")
        
        # Rysuj panel
        panel_height = 25 + len(metrics_text) * 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        for i, text in enumerate(metrics_text):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 25
        
        # Pasek postępu
        bar_width = frame_width - 40
        bar_x, bar_y = 20, frame_height - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        fill_width = int((progress / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 2)
        
        return frame

    def _ffmpeg_extract_segment(self, input_path: str, output_path: str,
                                start_time: float, end_time: float) -> bool:
        """
        Wytnij fragment wideo FFmpeg-iem tak, żeby wyglądał DOKŁADNIE jak w odtwarzaczu.

        Kluczowe:
        - NIE używamy -noautorotate
        - NIE kasujemy metadanych rotacji
        - FFmpeg sam stosuje rotację tak jak odtwarzacze
        """

        quality_map = {'low': '28', 'medium': '23', 'high': '18', 'max': '15'}
        crf = quality_map.get(self.video_quality, '23')

        try:
            cmd = [
                'ffmpeg',
                '-y',
                '-ss', f'{start_time:.3f}',
                '-to', f'{end_time:.3f}',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', crf,
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-c:a', 'aac',
                '-ac', '2',
                '-ar', '48000',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                if self.debug:
                    print("⚠️ FFmpeg extract error:", result.stderr)
                return False

            return True

        except Exception as e:
            if self.debug:
                print(f"⚠️  Błąd FFmpeg (_ffmpeg_extract_segment): {e}")
            return False

    
    def _convert_to_h264(self, input_path: str, output_path: str) -> bool:
        """
        Prosta konwersja do H.264 bez kombinowania z metadanymi rotacji.
        Używana tylko w fallbacku (gdy nie uda się wyciąć segmentu FFmpeg-iem).
        """
        try:
            quality_map = {'low': '28', 'medium': '23', 'high': '18', 'max': '15'}
            crf = quality_map.get(self.video_quality, '23')

            cmd = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', crf,
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            if os.path.exists(input_path):
                os.remove(input_path)

            return True

        except Exception as e:
            if self.debug:
                print(f"⚠️  Błąd konwersji H.264: {e}")
            # w ostateczności zostaw AVI
            return False

    
    def _create_visualization(
        self,
        signals: Dict[str, np.ndarray],
        repetitions: List[Dict],
        output_path: Path,
        person_id: str,
        video_id: str
    ):
        """Stwórz wizualizację dla wideo"""
        
        if self.exercise_type == 'pushup':
            # Wybierz kluczowe sygnały zależnie od orientacji
            camera_orientation = signals.get('camera_orientation', 'unknown')
            
            if camera_orientation == 'side':
                key_signals = ['avg_hip_y', 'avg_shoulder_y', 'chest_ground_distance', 
                              'avg_elbow_angle', 'nose_y', 'hip_velocity']
            elif camera_orientation == 'front':
                key_signals = ['chest_ground_distance', 'avg_shoulder_y', 'shoulder_width',
                              'elbow_width', 'avg_elbow_angle', 'wrist_shoulder_distance']
            elif camera_orientation == 'angle':
                key_signals = ['avg_shoulder_y', 'chest_ground_distance', 'avg_hip_y',
                              'shoulder_hip_distance', 'avg_elbow_angle', 'elbow_knee_distance']
            else:
                key_signals = ['avg_shoulder_y', 'avg_hip_y', 'chest_ground_distance',
                              'avg_elbow_angle', 'shoulder_hip_distance', 'hip_velocity']
        else:  # squat
            key_signals = [
                'avg_hip_y', 'avg_knee_angle', 'hip_knee_distance',
                'avg_hip_angle', 'torso_angle', 'avg_heel_y', 'heel_elevation'
            ]
        
        available_signals = {k: v for k, v in signals.items() if k in key_signals}
        
        if not available_signals:
            return
        
        n_signals = len(available_signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(16, 3 * n_signals))
        
        if n_signals == 1:
            axes = [axes]
        
        for idx, (signal_name, signal_data) in enumerate(available_signals.items()):
            ax = axes[idx]
            time = np.arange(len(signal_data)) / self.fps
            
            ax.plot(time, signal_data, label=signal_name, alpha=0.7, linewidth=2, color='steelblue')
            
            for rep in repetitions:
                is_edge = rep.get('note') in ['edge_start', 'edge_end']
                
                if self.exercise_type == 'squat':
                    depth_cat = rep.get('depth_category', 'unknown')
                    if depth_cat == 'deep':
                        color = 'darkgreen'
                    elif depth_cat == 'parallel':
                        color = 'green'
                    elif depth_cat == 'shallow':
                        color = 'yellow'
                    else:
                        color = 'gray'
                    if is_edge:
                        color = 'orange'
                else:
                    color = 'yellow' if is_edge else 'green'
                
                alpha = 0.15 if is_edge else 0.25
                
                ax.axvspan(rep['start_time'], rep['end_time'], alpha=alpha, color=color)
                
                mid_time = (rep['start_time'] + rep['end_time']) / 2
                y_pos = ax.get_ylim()[1] * 0.95
                
                label = f"#{rep['rep_id']}"
                if 'quality_score' in rep:
                    label += f"\nQ:{rep['quality_score']:.2f}"
                if self.exercise_type == 'squat' and 'depth_category' in rep:
                    label += f"\n{rep['depth_category']}"
                
                bbox_props = dict(
                    boxstyle='round,pad=0.4',
                    facecolor='yellow' if is_edge else 'lightgreen',
                    alpha=0.7,
                    edgecolor='orange' if is_edge else 'darkgreen',
                    linewidth=1.5
                )
                
                ax.text(mid_time, y_pos, label, ha='center', va='top',
                       fontsize=8, fontweight='bold', bbox=bbox_props)
            
            ax.set_xlabel('Time(s)', fontsize=11)
            ax.set_ylabel(signal_name, fontsize=11)
            ax.set_title(f'{signal_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        exercise_name = "Pompki" if self.exercise_type == 'pushup' else "Przysiady"
        plt.suptitle(f'Analiza - {exercise_name} - {person_id} / {video_id}', fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
    
    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Oblicz kąt"""
        a = np.array([p1['x'], p1['y']])
        b = np.array([p2['x'], p2['y']])
        c = np.array([p3['x'], p3['y']])
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _interpolate_nans(self, signal: np.ndarray) -> np.ndarray:
        """Interpoluj NaN"""
        nans = np.isnan(signal)
        if nans.all():
            return signal
        
        x = np.arange(len(signal))
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
        return signal