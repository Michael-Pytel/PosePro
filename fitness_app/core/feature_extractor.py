import pandas as pd
from fitness_app.metrics.report_repetition_tools_basic import get_rom
from fitness_app.metrics.report_repetition_tools_plank import get_plank
from fitness_app.metrics.report_repetition_tools_head import get_head

class FeatureExtractor:
    """
    Responsible for converting raw time-series signals and repetition metadata
    into the exact feature DataFrames expected by the ML models.
    """

    def extract_features(self, signals, visibility_scores, rep, fps):
        """
        Main entry point. Returns a dictionary containing the feature DataFrames
        for all available criteria (rom, hips, head).
        """
        # 1. Compute raw dictionary metrics (using your existing utils)
        raw_rom = get_rom(signals, visibility_scores, rep, fps)
        raw_plank = get_plank(signals, visibility_scores, rep, fps)
        raw_head = get_head(signals, visibility_scores, rep)
        
        ui_features = {
            'rom': raw_rom,
            'hips': raw_plank,
            'head': raw_head
        }

        # 2. Transform into model-ready DataFrames
        model_inputs = {
            'rom': self._prepare_rom_features(raw_rom),
            'hips': self._prepare_hips_features(raw_plank),
            'head': self._prepare_head_features(raw_head)
        }

        return ui_features, model_inputs

    def _prepare_rom_features(self, rom_data):
        row = {
            'max_elbow_angle': rom_data['max_elbow_angle'],
            'min_elbow_angle': rom_data['min_elbow_angle'],
            'mean_elbow_angle': rom_data['mean_elbow_angle'],
            'elbow_angle_q_10': rom_data['elbow_angle_q_10'],
            'elbow_angle_q_25': rom_data['elbow_angle_q_25'],
            'elbow_angle_q_50': rom_data['elbow_angle_q_50'],
            'elbow_angle_q_75': rom_data['elbow_angle_q_75'],
            'elbow_angle_q_90': rom_data['elbow_angle_q_90'],
            'elbow_angle_std': rom_data['elbow_angle_std'],
            
            # Critical renaming & casting matches training pipeline
            'range_of_motion_angle': rom_data['range_of_motion'], 
            'full_depth': int(rom_data['full_depth']),            
            'full_lockout': int(rom_data['full_lockout']),        
        }
        return pd.DataFrame([row])

    def _prepare_hips_features(self, plank_data):
        row = {
            # Hip Deviations
            'hip_deviation_bottom': plank_data['hip_deviation_bottom'],
            'hip_deviation_max_sag': plank_data['hip_deviation_max_sag'],
            'hip_deviation_max_pike': plank_data['hip_deviation_max_pike'],
            'hip_deviation_mean_abs': plank_data['hip_deviation_mean_abs'],
            'hip_deviation_std': plank_data['hip_deviation_std'],
            'hip_deviation_q_10': plank_data['hip_deviation_q_10'],
            'hip_deviation_q_25': plank_data['hip_deviation_q_25'],
            'hip_deviation_q_50': plank_data['hip_deviation_q_50'],
            'hip_deviation_q_75': plank_data['hip_deviation_q_75'],
            'hip_deviation_q_90': plank_data['hip_deviation_q_90'],

            # Torso Angles
            'torso_angle_bottom_deg': plank_data['torso_angle_bottom_deg'],
            'torso_angle_max_deg': plank_data['torso_angle_max_deg'],
            'torso_angle_min_deg': plank_data['torso_angle_min_deg'],
            'torso_angle_mean_deg': plank_data['torso_angle_mean_deg'],
            'torso_angle_range_deg': plank_data['torso_angle_range_deg'],
            'torso_angle_q_10_deg': plank_data['torso_angle_q_10_deg'],
            'torso_angle_q_25_deg': plank_data['torso_angle_q_25_deg'],
            'torso_angle_q_50_deg': plank_data['torso_angle_q_50_deg'],
            'torso_angle_q_75_deg': plank_data['torso_angle_q_75_deg'],
            'torso_angle_q_90_deg': plank_data['torso_angle_q_90_deg'],
            'torso_angle_std': plank_data['torso_angle_std'],
            
            # Body Angles
            'body_angle_bottom_deg': plank_data['body_angle_bottom_deg'],
            'body_angle_max_deg': plank_data['body_angle_max_deg'],
            'body_angle_min_deg': plank_data['body_angle_min_deg'],
            'body_angle_mean_deg': plank_data['body_angle_mean_deg'],
            'body_angle_range_deg': plank_data['body_angle_range_deg'],
            'body_angle_q_10_deg': plank_data['body_angle_q_10_deg'],
            'body_angle_q_25_deg': plank_data['body_angle_q_25_deg'],
            'body_angle_q_50_deg': plank_data['body_angle_q_50_deg'],
            'body_angle_q_75_deg': plank_data['body_angle_q_75_deg'],
            'body_angle_q_90_deg': plank_data['body_angle_q_90_deg'],
            'body_angle_std': plank_data['body_angle_std']
        }
        return pd.DataFrame([row])

    def _prepare_head_features(self, head_data):
        row = {
            # Torso-Head Angle
            'torso_head_angle_bottom': head_data['torso_head_angle_bottom'],
            'torso_head_angle_min': head_data['torso_head_angle_min'],
            'torso_head_angle_max': head_data['torso_head_angle_max'],
            'torso_head_angle_range': head_data['torso_head_angle_range'],
            'torso_head_angle_mean': head_data['torso_head_angle_mean'],
            'torso_head_angle_q_10': head_data['torso_head_angle_q_10'],
            'torso_head_angle_q_25': head_data['torso_head_angle_q_25'],
            'torso_head_angle_q_50': head_data['torso_head_angle_q_50'],
            'torso_head_angle_q_75': head_data['torso_head_angle_q_75'],
            'torso_head_angle_q_90': head_data['torso_head_angle_q_90'],
            'torso_head_angle_std': head_data['torso_head_angle_std'],

            # Head Forward Norm
            'head_forward_norm_bottom': head_data['head_forward_norm_bottom'],
            'head_forward_norm_min': head_data['head_forward_norm_min'],
            'head_forward_norm_max': head_data['head_forward_norm_max'],
            'head_forward_norm_range': head_data['head_forward_norm_range'],
            'head_forward_norm_mean': head_data['head_forward_norm_mean'],
            'head_forward_norm_q_10': head_data['head_forward_norm_q_10'],
            'head_forward_norm_q_25': head_data['head_forward_norm_q_25'],
            'head_forward_norm_q_50': head_data['head_forward_norm_q_50'],
            'head_forward_norm_q_75': head_data['head_forward_norm_q_75'],
            'head_forward_norm_q_90': head_data['head_forward_norm_q_90'],
            'head_forward_norm_std': head_data['head_forward_norm_std'],

            # Head Tilt
            'head_tilt_angle_bottom': head_data['head_tilt_angle_bottom'],
            'head_tilt_angle_min': head_data['head_tilt_angle_min'],
            'head_tilt_angle_max': head_data['head_tilt_angle_max'],
            'head_tilt_angle_range': head_data['head_tilt_angle_range'],
            'head_tilt_angle_mean': head_data['head_tilt_angle_mean'],
            'head_tilt_angle_q_10': head_data['head_tilt_angle_q_10'],
            'head_tilt_angle_q_25': head_data['head_tilt_angle_q_25'],
            'head_tilt_angle_q_50': head_data['head_tilt_angle_q_50'],
            'head_tilt_angle_q_75': head_data['head_tilt_angle_q_75'],
            'head_tilt_angle_q_90': head_data['head_tilt_angle_q_90'],
            'head_tilt_angle_std': head_data['head_tilt_angle_std']
        }
        return pd.DataFrame([row])