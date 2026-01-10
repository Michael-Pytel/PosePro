from fitness_app.apps import FitnessAppConfig

class Predictor:
    """
    Handles loading (via AppConfig) and querying of ML models.
    """
    
    def predict_repetition(self, feature_dataframes):
        """
        Takes a dict of feature dataframes (output from FeatureExtractor)
        and returns a dict of predictions.
        """
        predictions = {}
        
        # 1. Elbow/ROM Model
        predictions['rom'] = self._predict_single(
            feature_dataframes.get('rom'), 
            model_key='rom'
        )

        # 2. Hips Model
        predictions['hips'] = self._predict_single(
            feature_dataframes.get('hips'), 
            model_key='hips'
        )

        # 3. Head Model
        predictions['head'] = self._predict_single(
            feature_dataframes.get('head'), 
            model_key='head'
        )
        
        return predictions

    def _predict_single(self, df_features, model_key):
        """Helper to safely predict a single criterion."""
        
        # Safety check: Do we have features and is the model loaded?
        if df_features is None or model_key not in FitnessAppConfig.models:
            return None

        try:
            model = FitnessAppConfig.models[model_key]
            
            # Predict Class (0 or 1)
            pred_class = int(model.predict(df_features)[0])
            
            # Predict Confidence (Probability)
            try:
                # Get probability of the predicted class
                probs = model.predict_proba(df_features)[0]
                confidence = float(probs.max())
            except AttributeError:
                # Fallback for models without predict_proba (e.g. some SVM configs)
                confidence = 1.0 

            return {
                'class': pred_class,
                'confidence': confidence,
                'is_correct': bool(pred_class == 0)
            }
            
        except Exception as e:
            print(f"Error predicting {model_key}: {e}")
            return None