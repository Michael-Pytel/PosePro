"""
Uniwersalny skrypt CLI do przetwarzania folderów z nagraniami ćwiczeń

UŻYCIE:
  python run_exercise_processor.py -i <folder> -o <output> -p <person_id> -e <exercise_type>

TYPY ĆWICZEŃ:
  pushup  - Pompki
  squat   - Przysiady

PRZYKŁADY:
  # Pompki
  python run_exercise_processor.py -i ./videos/pushups -o ./output -p athlete001 -e pushup
  
  # Przysiady
  python run_exercise_processor.py -i ./videos/squats -o ./output -p athlete001 -e squat
  
  # Z niestandardowymi parametrami
  python run_exercise_processor.py -i ./videos -o ./output -p person123 -e pushup --min-duration 0.8 --quality high --debug
"""

from exercise_batch_processor_2 import ExerciseBatchProcessor
import sys
import os
import argparse
import io
import subprocess

# Ustaw kodowanie UTF-8 dla stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def parse_arguments():
    """Parse argumenty z linii poleceń"""
    parser = argparse.ArgumentParser(
        description='🏋️ Uniwersalny batch processor do segmentacji ćwiczeń',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Pompki
  %(prog)s -i ./videos/pushups -o ./output -p athlete001 -e pushup
  
  # Przysiady  
  %(prog)s -i ./videos/squats -o ./output -p athlete001 -e squat
  
  # Z debugiem
  %(prog)s -i ./videos -o ./output -p person123 -e squat --debug --no-confirm
        """
    )
    
    # Wymagane argumenty
    parser.add_argument('-i', '--input', required=True,
                       help='Folder z nagraniami wejściowymi (WYMAGANE)')
    parser.add_argument('-o', '--output', required=True,
                       help='Folder docelowy dla wyników (WYMAGANE)')
    parser.add_argument('-p', '--person-id', required=True,
                       help='ID osoby, np. athlete001 (WYMAGANE)')
    parser.add_argument('-e', '--exercise', required=True,
                       choices=['pushup', 'squat'],
                       help='Typ ćwiczenia: pushup (pompki) lub squat (przysiady) (WYMAGANE)')
    
    # Opcjonalne parametry
    parser.add_argument('--min-duration', type=float, default=None,
                       help='Minimalny czas powtórzenia (s). Domyślnie: 0.6 dla pompek, 0.8 dla przysiadów')
    parser.add_argument('--max-duration', type=float, default=None,
                       help='Maksymalny czas powtórzenia (s). Domyślnie: 6.0 dla pompek, 8.0 dla przysiadów')
    parser.add_argument('--quality', choices=['low', 'medium', 'high', 'max'], default='medium',
                       help='Jakość wideo wyjściowego (domyślnie: medium)')
    parser.add_argument('--debug', action='store_true',
                       help='Tryb debug - więcej logów')
    parser.add_argument('--no-confirm', action='store_true',
                       help='Pomiń potwierdzenie - rozpocznij od razu')
    parser.add_argument('--visualize', action='store_true',  # ⭐ DODANE
                       help='Nakładaj punkty MediaPipe i metryki na klipy wideo')
    return parser.parse_args()

def main():
    # Emoji dla ćwiczeń
    EXERCISE_EMOJI = {
        'pushup': '💪',
        'squat': '🏋️'
    }
    
    EXERCISE_NAME = {
        'pushup': 'POMPEK',
        'squat': 'PRZYSIADÓW'
    }
    
    print("\n" + "="*80)
    print(f"🏋️ BATCH PROCESSOR - SEGMENTACJA ĆWICZEŃ")
    print("="*80 + "\n")
    
    # Parse argumenty
    args = parse_arguments()
    
    # KONFIGURACJA z argumentów
    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output
    PERSON_ID = args.person_id
    EXERCISE_TYPE = args.exercise
    
    # Parametry opcjonalne
    MIN_REP_DURATION = args.min_duration
    MAX_REP_DURATION = args.max_duration
    VIDEO_QUALITY = args.quality
    DEBUG = args.debug
    NO_CONFIRM = args.no_confirm
    VISUALIZE = args.visualize
    
    # Sprawdź czy folder istnieje
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ BŁĄD: Folder nie istnieje: {INPUT_FOLDER}")
        print(f"\n💡 Wskazówka:")
        print(f"   1. Utwórz folder: {INPUT_FOLDER}")
        print(f"   2. Umieść w nim pliki wideo (.mp4, .avi, .mov)")
        print(f"   3. Uruchom ponownie skrypt\n")
        sys.exit(1)
    
    # Wyświetl konfigurację
    emoji = EXERCISE_EMOJI.get(EXERCISE_TYPE, '🏋️')
    exercise_name = EXERCISE_NAME.get(EXERCISE_TYPE, EXERCISE_TYPE.upper())
    
    print(f"{emoji} Typ ćwiczenia: {exercise_name}")
    print()
    print("⚙️ Konfiguracja:")
    print(f"   📁 Folder wejściowy: {os.path.abspath(INPUT_FOLDER)}")
    print(f"   📂 Folder wyjściowy: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"   👤 Person ID: {PERSON_ID}")
    
    # Wyświetl czasy
    if MIN_REP_DURATION is None:
        min_time = 0.6 if EXERCISE_TYPE == 'pushup' else 0.8
        print(f"   ⏱️  Min czas: {min_time}s (domyślny)")
    else:
        print(f"   ⏱️  Min czas: {MIN_REP_DURATION}s")
    
    if MAX_REP_DURATION is None:
        max_time = 6.0 if EXERCISE_TYPE == 'pushup' else 8.0
        print(f"   ⏱️  Max czas: {max_time}s (domyślny)")
    else:
        print(f"   ⏱️  Max czas: {MAX_REP_DURATION}s")
    
    print(f"   🎬 Jakość wideo: {VIDEO_QUALITY}")
    print(f"   🐛 Debug: {'TAK' if DEBUG else 'NIE'}")
    print(f"   👁️  Wizualizacja landmarks: {'TAK' if VISUALIZE else 'NIE'}")  # ⭐ DODANE
    
    # ⭐ DODANE - ostrzeżenie o wydajności
    if VISUALIZE:
        print(f"\n   ⚠️  UWAGA: Wizualizacja wydłuża czas przetwarzania (~5x)!")
    
    print()
    
    # Potwierdź uruchomienie (chyba że --no-confirm)
    if not NO_CONFIRM:
        response = input(f"🚀 Rozpocząć przetwarzanie {exercise_name}? (tak/nie): ").strip().lower()
        
        if response not in ['tak', 't', 'yes', 'y']:
            print("❌ Anulowano")
            sys.exit(0)
    
    print("\n" + "="*80)
    print(f"▶️  ROZPOCZYNAM PRZETWARZANIE {exercise_name}...")
    print("="*80 + "\n")
    
    # Uruchom processor
    try:
        processor = ExerciseBatchProcessor(
            exercise_type=EXERCISE_TYPE,
            min_rep_duration=MIN_REP_DURATION,
            max_rep_duration=MAX_REP_DURATION,
            peak_prominence=0.08,
            video_quality=VIDEO_QUALITY,
            debug=DEBUG,
            visualize_landmarks=VISUALIZE
        )
        
        df_results = processor.process_folder(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            person_id=PERSON_ID,
            video_extensions=['.mp4', '.avi', '.mov', '.MOV', '.MP4']
        )
        
        # Podsumowanie
        if not df_results.empty:
            print("\n" + "="*80)
            print(f"✅ PRZETWARZANIE {exercise_name} ZAKOŃCZONE SUKCESEM!")
            print("="*80)
            
            print(f"\n📊 Wyniki:")
            print(f"   • Przetworzone wideo: {df_results['video_id'].nunique()}")
            print(f"   • Wykryte powtórzenia: {len(df_results)}")
            print(f"   • Średnia jakość: {df_results['quality_score'].mean():.3f}")
            print(f"   • Średni czas: {df_results['duration'].mean():.2f}s")
            
            # Statystyki specyficzne dla przysiadów
            if EXERCISE_TYPE == 'squat':
                print(f"\n🏋️  Rozkład głębokości:")
                depth_counts = df_results['depth_category'].value_counts()
                for depth, count in depth_counts.items():
                    emoji_depth = {
                        'deep': '🟢',
                        'parallel': '🟡',
                        'shallow': '🟠',
                        'unknown': '⚪'
                    }.get(depth, '⚪')
                    print(f"   {emoji_depth} {depth}: {count}")
                
                print(f"\n📐 Minimalny kąt kolana:")
                print(f"   • Średnia: {df_results['min_knee_angle'].mean():.1f}°")
                print(f"   • Min: {df_results['min_knee_angle'].min():.1f}°")
                print(f"   • Max: {df_results['min_knee_angle'].max():.1f}°")
            
            print(f"\n📁 Pliki wyjściowe:")
            print(f"   • Klipy wideo: {OUTPUT_FOLDER}/videos/")
            print(f"   • Wizualizacje: {OUTPUT_FOLDER}/visualizations/")
            print(f"   • Dane CSV: {OUTPUT_FOLDER}/repetitions_data.csv")
            
            # Ostrzeżenia o niskiej jakości
            low_quality = df_results[df_results['quality_score'] < 0.5]
            if not low_quality.empty:
                print(f"\n⚠️  Uwaga:")
                print(f"   • {len(low_quality)} powtórzeń ma jakość < 0.5")
                print(f"   • Sprawdź je w CSV: {OUTPUT_FOLDER}/repetitions_data.csv")
            
            print("\n" + "="*80)
            print(f"🎉 Gotowe! Sprawdź wyniki w folderze:")
            print(f"   {os.path.abspath(OUTPUT_FOLDER)}")
            print("="*80 + "\n")
            
            # Szybkie statystyki
            if DEBUG:
                print(f"\n📈 SZCZEGÓŁOWE STATYSTYKI:")
                print(df_results[['quality_score', 'duration', 'symmetry_score']].describe())
            
            # ⭐ NOWE: Automatycznie uruchom ekstrakcję landmarks z videos_nooverlay
            print("\n" + "="*80)
            print("🔍 Ekstrakcja landmarks z videos_nooverlay...")
            print("="*80 + "\n")
            
            try:
                # Ścieżka do skryptu ekstrakcji
                extraction_script = os.path.join(
                    os.path.dirname(__file__),
                    'extract_landmarks_multi.py'
                )
                
                # Ścieżki do folderów
                videos_nooverlay = os.path.join(OUTPUT_FOLDER, 'videos_nooverlay')
                output_landmarks = os.path.join(OUTPUT_FOLDER, 'landmarks_output')
                
                if os.path.exists(videos_nooverlay):
                    print(f"📁 Input:  {videos_nooverlay}")
                    print(f"📂 Output: {output_landmarks}\n")
                    
                    # Uruchom skrypt ekstrakcji landmarks
                    extraction_cmd = [
                        sys.executable,
                        extraction_script,
                        '-i', videos_nooverlay,
                        '-o', output_landmarks,
                        '--model', 'full',
                        '--workers', '4'
                    ]
                    
                    # Set unbuffered output
                    env = os.environ.copy()
                    env['PYTHONUNBUFFERED'] = '1'
                    
                    result = subprocess.run(extraction_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
                    
                    # Print subprocess output
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)
                    
                    if result.returncode == 0:
                        print(f"\n✅ Ekstrakcja landmarks ZAKOŃCZONA!")
                        
                        # ⭐ NOWE: Automatycznie uruchom obliczanie kątów
                        print("\n" + "="*80)
                        print("📐 Obliczanie kątów joints...")
                        print("="*80 + "\n")
                        
                        try:
                            # Ścieżka do skryptu obliczania kątów
                            angle_script = os.path.join(
                                os.path.dirname(__file__),
                                'angle_calculator_batch.py'
                            )
                            
                            # Ścieżki do folderów
                            landmarks_angles = os.path.join(OUTPUT_FOLDER, 'landmarks_angles')
                            
                            print(f"📁 Input:  {output_landmarks}")
                            print(f"📂 Output: {landmarks_angles}\n")
                            
                            # Uruchom skrypt obliczania kątów
                            angles_cmd = [
                                sys.executable,
                                angle_script,
                                '-i', output_landmarks,
                                '-o', landmarks_angles,
                                '--min-visibility', '0.5'
                            ]
                            
                            # Set unbuffered output
                            env = os.environ.copy()
                            env['PYTHONUNBUFFERED'] = '1'
                            
                            result_angles = subprocess.run(angles_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
                            
                            # Print subprocess output
                            if result_angles.stdout:
                                print(result_angles.stdout)
                            if result_angles.stderr:
                                print("STDERR:", result_angles.stderr)
                            
                            if result_angles.returncode == 0:
                                print(f"\n✅ Obliczanie kątów ZAKOŃCZONE!")
                                
                                # ⭐ NOWE: Automatycznie uruchom agregację kątów
                                print("\n" + "="*80)
                                print("📊 Agregacja kątów do reps.csv...")
                                print("="*80 + "\n")
                                
                                try:
                                    # Ścieżka do skryptu agregacji
                                    aggregation_script = os.path.join(
                                        os.path.dirname(__file__),
                                        'aggregate_angles2.py'
                                    )
                                    
                                    # Ścieżki
                                    reps_output = os.path.join(
                                        os.path.dirname(OUTPUT_FOLDER),
                                        'reps.csv'
                                    )
                                    
                                    print(f"📁 Input:  {landmarks_angles}")
                                    print(f"📂 Output: {reps_output}\n")
                                    
                                    # Uruchom skrypt agregacji
                                    aggregation_cmd = [
                                        sys.executable,
                                        aggregation_script,
                                        '-i', landmarks_angles,
                                        '-o', reps_output,
                                        '--valid-threshold', '0.5'
                                    ]
                                    
                                    # Set unbuffered output
                                    env = os.environ.copy()
                                    env['PYTHONUNBUFFERED'] = '1'
                                    
                                    result_agg = subprocess.run(aggregation_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
                                    
                                    # Print subprocess output
                                    if result_agg.stdout:
                                        print(result_agg.stdout)
                                    if result_agg.stderr:
                                        print("STDERR:", result_agg.stderr)
                                    
                                    if result_agg.returncode == 0:
                                        print(f"\n✅ Agregacja kątów ZAKOŃCZONA!")
                                        print(f"📄 Wynik: {reps_output}")
                                        
                                        # ⭐ NOWE: Automatycznie uruchom predykcje modelu
                                        print("\n" + "="*80)
                                        print("🔮 Predykcja modelu na podstawie kątów...")
                                        print("="*80 + "\n")
                                        
                                        try:
                                            # Ścieżka do skryptu predykcji
                                            prediction_script = os.path.join(
                                                os.path.dirname(__file__),
                                                'predict_model.py'
                                            )
                                            
                                            # Ścieżka wyjściowa predykcji
                                            predictions_output = os.path.join(
                                                os.path.dirname(OUTPUT_FOLDER),
                                                'reps_predictions.csv'
                                            )
                                            
                                            print(f"📁 Input:  {reps_output}")
                                            print(f"📂 Output: {predictions_output}\n")
                                            
                                            # Uruchom skrypt predykcji
                                            prediction_cmd = [
                                                sys.executable,
                                                prediction_script,
                                                '-i', reps_output,
                                                '-o', predictions_output,
                                                '-m', os.path.join(
                                                    os.path.dirname(__file__),
                                                    '../models'
                                                )
                                            ]
                                            
                                            # Set unbuffered output
                                            env = os.environ.copy()
                                            env['PYTHONUNBUFFERED'] = '1'
                                            
                                            result_pred = subprocess.run(prediction_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
                                            
                                            # Print subprocess output
                                            if result_pred.stdout:
                                                print(result_pred.stdout)
                                            if result_pred.stderr:
                                                print("STDERR:", result_pred.stderr)
                                            
                                            if result_pred.returncode == 0:
                                                print(f"\n✅ Predykcja modelu ZAKOŃCZONA!")
                                                print(f"📄 Wynik: {predictions_output}")
                                            else:
                                                print(f"\n⚠️  Predykcja modelu zakończyła się z kodem: {result_pred.returncode}")
                                        
                                        except Exception as e:
                                            print(f"\n⚠️  Błąd podczas predykcji modelu: {e}")
                                            if DEBUG:
                                                import traceback
                                                traceback.print_exc()
                                    else:
                                        print(f"\n⚠️  Agregacja kątów zakończyła się z kodem: {result_agg.returncode}")
                                
                                except Exception as e:
                                    print(f"\n⚠️  Błąd podczas agregacji kątów: {e}")
                                    if DEBUG:
                                        import traceback
                                        traceback.print_exc()
                            else:
                                print(f"\n⚠️  Obliczanie kątów zakończyło się z kodem: {result_angles.returncode}")
                        
                        except Exception as e:
                            print(f"\n⚠️  Błąd podczas obliczania kątów: {e}")
                            if DEBUG:
                                import traceback
                                traceback.print_exc()
                    else:
                        print(f"\n⚠️  Ekstrakcja landmarks zakończyła się z kodem: {result.returncode}")
                else:
                    print(f"⚠️  Folder videos_nooverlay nie istnieje: {videos_nooverlay}")
            
            except Exception as e:
                print(f"\n⚠️  Błąd podczas ekstrakcji landmarks: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
        
        else:
            print(f"\n❌ Nie udało się przetworzyć żadnych nagrań {exercise_name}")
            print("💡 Sprawdź czy w folderze są pliki wideo (.mp4, .avi, .mov)\n")
        
    except ValueError as e:
        print(f"\n❌ BŁĄD KONFIGURACJI: {e}")
        print("\n💡 Dozwolone typy ćwiczeń: pushup, squat\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ BŁĄD podczas przetwarzania: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()