import pandas as pd
import numpy as np
import pickle
import joblib
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """Load and use pkl models to make predictions on angle features"""
    
    def __init__(self, models_dir, verbose=True):
        """
        Initialize predictor by loading all models
        
        Args:
            models_dir: Path to directory containing .pkl models
            verbose: Whether to print loading information
        """
        self.models_dir = Path(models_dir)
        self.verbose = verbose
        self.models = {}
        self.feature_names = []
        self.target_columns = []
        
        # Load feature names and target columns
        self._load_metadata()
        
        # Load all models
        self._load_models()
    
    def _load_metadata(self):
        """Load feature names and target columns from txt files"""
        # Load feature names
        feature_file = self.models_dir / 'feature_names.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines() if line.strip()]
            if self.verbose:
                logger.info(f"✓ Loaded {len(self.feature_names)} feature names")
        else:
            logger.warning(f"Feature names file not found: {feature_file}")
        
        # Load target columns
        target_file = self.models_dir / 'target_columns.txt'
        if target_file.exists():
            with open(target_file, 'r') as f:
                self.target_columns = [line.strip() for line in f.readlines() if line.strip()]
            if self.verbose:
                logger.info(f"✓ Loaded {len(self.target_columns)} target columns: {self.target_columns}")
        else:
            logger.warning(f"Target columns file not found: {target_file}")
    
    def _load_models(self):
        """Load all .pkl model files from models directory"""
        pkl_files = list(self.models_dir.glob('*.pkl'))
        
        if not pkl_files:
            logger.warning(f"No .pkl files found in {self.models_dir}")
            return
        
        for pkl_file in pkl_files:
            try:
                # Try pickle first
                with open(pkl_file, 'rb') as f:
                    try:
                        model = pickle.load(f)
                    except Exception as e_pickle:
                        # Try joblib as a fallback
                        try:
                            model = joblib.load(pkl_file)
                        except Exception:
                            # Final fallback: attempt custom Unpickler mapping
                            try:
                                f.seek(0)
                                import pickle as _pickle
                                from sklearn.ensemble import RandomForestClassifier

                                class MappingUnpickler(_pickle.Unpickler):
                                    def find_class(self, module, name):
                                        # Handle cases where the pickle references a module named
                                        # 'RandomForestClassifier' (or similar mis-serialized globals)
                                        if module == 'RandomForestClassifier':
                                            return RandomForestClassifier
                                        return super().find_class(module, name)

                                model = MappingUnpickler(f).load()
                            except Exception as e_final:
                                raise e_final from e_pickle
                
                model_name = pkl_file.stem
                self.models[model_name] = model
                
                if self.verbose:
                    logger.info(f"✓ Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"✗ Failed to load {pkl_file.name}: {str(e)}")
    
    def _prepare_features(self, df):
        """
        Prepare features for prediction, matching with expected feature names
        
        Args:
            df: DataFrame with angle features
        
        Returns:
            numpy array with features in correct order, dict with missing features
        """
        missing_features = []
        feature_vectors = []
        
        for feat_name in self.feature_names:
            if feat_name in df.columns:
                feature_vectors.append(df[feat_name].values)
            else:
                missing_features.append(feat_name)
                # Use NaN for missing features
                feature_vectors.append(np.full(len(df), np.nan))
        
        if missing_features and self.verbose:
            logger.warning(f"⚠️  Missing {len(missing_features)} features: {missing_features[:5]}...")
        
        # Stack features as columns
        X = np.column_stack(feature_vectors)
        
        return X, missing_features
    
    def predict(self, csv_path):
        """
        Make predictions on data from CSV file
        
        Args:
            csv_path: Path to reps.csv file
        
        Returns:
            dict with predictions for each model
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔮 MAKING PREDICTIONS")
        logger.info(f"{'='*80}")
        logger.info(f"Input file: {csv_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
        except Exception as e:
            logger.error(f"✗ Failed to load CSV: {str(e)}")
            return None
        
        # Prepare features
        logger.info(f"\nPreparing features...")
        X, missing_features = self._prepare_features(df)
        logger.info(f"✓ Feature matrix shape: {X.shape}")
        
        if missing_features:
            logger.warning(f"⚠️  {len(missing_features)} features missing")
        
        # Handle NaN values - fill with 0 or column mean
        logger.info(f"\nHandling NaN values...")
        nan_mask = np.isnan(X)
        if nan_mask.sum() > 0:
            logger.warning(f"⚠️  Found {nan_mask.sum()} NaN values in feature matrix")
            # Replace NaN with column mean, or 0 if all values are NaN
            for col_idx in range(X.shape[1]):
                col = X[:, col_idx]
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    X[np.isnan(col), col_idx] = 0
                else:
                    X[np.isnan(col), col_idx] = col_mean
        
        # Make predictions with each model
        predictions = {}
        logger.info(f"\n{'='*80}")
        logger.info(f"Running predictions with {len(self.models)} models...")
        logger.info(f"{'='*80}\n")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Predicting with {model_name}...")
                
                # Try to predict
                try:
                    y_pred = model.predict(X)
                except Exception as e:
                    logger.warning(f"  predict() failed: {str(e)}, trying predict_proba()...")
                    try:
                        y_pred = model.predict_proba(X)
                    except Exception as e2:
                        logger.error(f"  Both predict() and predict_proba() failed: {str(e2)}")
                        continue
                
                predictions[model_name] = y_pred
                logger.info(f"✓ {model_name}: predictions shape {y_pred.shape}")
                
                # Log sample predictions
                if len(y_pred) > 0:
                    logger.info(f"  Sample predictions (first 3):")
                    for idx in range(min(3, len(y_pred))):
                        logger.info(f"    Sample {idx}: {y_pred[idx]}")
            
            except Exception as e:
                logger.error(f"✗ Error with {model_name}: {str(e)}")
        
        # Add predictions to dataframe
        logger.info(f"\n{'='*80}")
        logger.info(f"Summary of all predictions:")
        logger.info(f"{'='*80}\n")
        
        df_with_predictions = df.copy()
        
        for model_name, y_pred in predictions.items():
            # Handle different prediction formats
            if len(y_pred.shape) == 1:
                # Single column output
                df_with_predictions[f'pred_{model_name}'] = y_pred
                logger.info(f"{model_name}:")
                logger.info(f"  Predictions per sample: 1")
                logger.info(f"  Unique values: {len(np.unique(y_pred))}")
                if len(np.unique(y_pred)) <= 10:
                    logger.info(f"  Values: {np.unique(y_pred)}")
            else:
                # Multi-column output (e.g., probabilities)
                for col_idx in range(y_pred.shape[1]):
                    df_with_predictions[f'pred_{model_name}_class{col_idx}'] = y_pred[:, col_idx]
                logger.info(f"{model_name}:")
                logger.info(f"  Predictions per sample: {y_pred.shape[1]}")
        
        logger.info(f"\n✓ Total new columns added: {len(predictions)}")
        
        return df_with_predictions
    
    def save_predictions(self, df_predictions, output_path):
        """Save predictions to CSV file"""
        try:
            df_predictions.to_csv(output_path, index=False)
            logger.info(f"\n✓ Predictions saved to: {output_path}")
            logger.info(f"  Shape: {df_predictions.shape}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save predictions: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained models on angle features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input CSV file with angle features (typically reps.csv)'
    )
    
    parser.add_argument(
        '-m', '--models',
        type=str,
        default='../exercise_processor/models',
        help='Path to directory containing .pkl models'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file for predictions (default: input_with_predictions.csv in same directory)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    models_path = Path(args.models)
    
    # Check if input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Check if models directory exists
    if not models_path.exists():
        logger.error(f"Models directory not found: {models_path}")
        return
    
    # Default output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
    else:
        output_path = Path(args.output)
    
    # Create predictor
    predictor = ModelPredictor(models_path, verbose=not args.quiet)
    
    if not predictor.models:
        logger.error("No models were loaded. Cannot proceed.")
        return
    
    # Make predictions
    df_predictions = predictor.predict(str(input_path))
    
    if df_predictions is not None:
        # Save predictions
        predictor.save_predictions(df_predictions, str(output_path))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ PREDICTION COMPLETE")
        logger.info(f"{'='*80}\n")
    else:
        logger.error("Prediction failed")

if __name__ == "__main__":
    main()
