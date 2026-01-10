import os
import joblib
from django.apps import AppConfig
from django.conf import settings

class FitnessAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fitness_app'

    # Dictionary to hold multiple models: {'elbow': pipeline_obj, 'shoulder': ...}
    models = {}

    def ready(self):
        """
        Runs once on server startup. Loads all body part models into memory.
        """
        # Define your mapping of { key: filename }
        model_files = {
            'hips': 'best_model_hips.pkl',
            'head': 'best_model_head.pkl',
            'rom': 'best_model_rom.pkl'
        }

        base_path = os.path.join(settings.BASE_DIR, 'fitness_app\\ml\\models\\')

        print("\n initializing ML Models...")
        
        for part, filename in model_files.items():
            path = os.path.join(base_path, filename)

            if os.path.exists(path):
                try:
                    # Load and store in the dictionary
                    FitnessAppConfig.models[part] = joblib.load(path)
                    print(f"   ✅ Loaded: {part.capitalize()} ({filename})")
                except Exception as e:
                    print(f"   ❌ Failed to load {part}: {e}")
            else:
                print(f"   ⚠️  File not found: {filename}")