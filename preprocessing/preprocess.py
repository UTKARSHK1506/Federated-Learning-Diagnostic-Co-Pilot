import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

# Canonical feature column names in exact required order
FEATURE_COLUMNS = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]
TARGET_COLUMN = 'cardio'

# Continuous features requiring z-score standardization
CONTINUOUS_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']


class ClinicDataPreprocessor:
    """
    Data Preprocessor for converting raw clinic CSV records into PyTorch tensors
    suitable for DiagnosticModel training, validation, testing, and inference.
    
    Ensures that StandardScaler is fitted ONLY on the local training split of a clinic,
    eliminating data leakage across validation/test sets or other clinics.
    """
    
    def __init__(self, scaler: Optional[StandardScaler] = None):
        """
        Initializes the preprocessor. If a pre-fitted scaler is provided, it will be used.
        """
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = scaler is not None

    def fit_transform_features(self, df_train: pd.DataFrame) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fits the StandardScaler ONLY on the continuous features of df_train and transforms it.
        
        Args:
            df_train (pd.DataFrame): Training subset dataframe containing feature columns and optional target.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (X_train_tensor, y_train_tensor)
        """
        # Ensure features exclude 'id' and follow exact canonical order
        features_df = df_train[FEATURE_COLUMNS].copy()
        
        # Fit scaler ONLY on continuous features of local training split
        continuous_scaled = self.scaler.fit_transform(features_df[CONTINUOUS_FEATURES])
        self.is_fitted = True
        
        processed_df = features_df.copy()
        processed_df[CONTINUOUS_FEATURES] = continuous_scaled
        
        # Convert to PyTorch float32 tensors
        X_tensor = torch.tensor(processed_df.values, dtype=torch.float32)
        y_tensor = None
        if TARGET_COLUMN in df_train.columns:
            y_tensor = torch.tensor(df_train[TARGET_COLUMN].values, dtype=torch.float32).unsqueeze(1)
            
        return X_tensor, y_tensor

    def transform_features(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Transforms validation, test, or new inference data using the already fitted scaler.
        
        Args:
            df (pd.DataFrame): Dataframe containing feature columns and optional target.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (X_tensor, y_tensor)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scaler must be fitted on training data before transform_features().")
            
        features_df = df[FEATURE_COLUMNS].copy()
        
        # Apply pre-fitted scaler WITHOUT refitting
        continuous_scaled = self.scaler.transform(features_df[CONTINUOUS_FEATURES])
        
        processed_df = features_df.copy()
        processed_df[CONTINUOUS_FEATURES] = continuous_scaled
        
        X_tensor = torch.tensor(processed_df.values, dtype=torch.float32)
        y_tensor = None
        if TARGET_COLUMN in df.columns:
            y_tensor = torch.tensor(df[TARGET_COLUMN].values, dtype=torch.float32).unsqueeze(1)
            
        return X_tensor, y_tensor


def load_and_preprocess_clinic(
    csv_path: str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Loads a clinic CSV file, performs a stratified train/val/test split (80/10/10), fits a local
    StandardScaler ONLY on the local training set, and transforms all splits into PyTorch tensors.
    
    Args:
        csv_path (str): Path to the clinic CSV file (e.g. 'data/clinic_A.csv').
        val_ratio (float): Ratio of total records reserved for validation (default: 0.10).
        test_ratio (float): Ratio of total records reserved for testing (default: 0.10).
        random_seed (int): Fixed seed for reproducible stratified splitting (default: 42).
        
    Returns:
        Tuple containing:
            - X_train (torch.Tensor): Shape (N_train, 11) float32
            - y_train (torch.Tensor): Shape (N_train, 1) float32
            - X_val (torch.Tensor): Shape (N_val, 11) float32
            - y_val (torch.Tensor): Shape (N_val, 1) float32
            - X_test (torch.Tensor): Shape (N_test, 11) float32
            - y_test (torch.Tensor): Shape (N_test, 1) float32
            - scaler (StandardScaler): The local scaler fitted ONLY on X_train.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinic dataset file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Stratified split: Train (80%) vs Temp (20%)
    temp_ratio = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df[TARGET_COLUMN],
        random_state=random_seed
    )
    
    # 2. Stratified split: Val (10%) vs Test (10%) from Temp (20%)
    relative_test_ratio = test_ratio / temp_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_ratio,
        stratify=df_temp[TARGET_COLUMN],
        random_state=random_seed
    )
    
    # 3. Instantiate local clinic preprocessor
    preprocessor = ClinicDataPreprocessor()
    
    # 4. Fit scaler ONLY on local training data and transform train set
    X_train, y_train = preprocessor.fit_transform_features(df_train)
    
    # 5. Transform validation and test sets using the SAME training-fitted scaler
    X_val, y_val = preprocessor.transform_features(df_val)
    X_test, y_test = preprocessor.transform_features(df_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor.scaler


def transform_new_patient(raw_patient_data: pd.DataFrame, scaler: StandardScaler) -> torch.Tensor:
    """
    Transforms new unscaled patient feature data using an already fitted local scaler (for application inference).
    
    Args:
        raw_patient_data (pd.DataFrame): Raw patient DataFrame containing the 11 feature columns.
        scaler (StandardScaler): Fitted local clinic scaler.
        
    Returns:
        torch.Tensor: Preprocessed feature tensor of shape (N, 11) float32.
    """
    preprocessor = ClinicDataPreprocessor(scaler=scaler)
    X_tensor, _ = preprocessor.transform_features(raw_patient_data)
    return X_tensor
