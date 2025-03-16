"""
Analytical Engine Module

Analyzes data to identify inefficiencies, cost overruns, profitability constraints, 
and customer dissatisfaction factors. Implements predictive analytics to forecast 
the impact of proposed changes.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/analytical_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnalyticalEngine")

class AnalyticalEngine:
    """
    Main class for the Analytical Engine module that processes and analyzes business data
    to identify optimization opportunities.
    """
    def __init__(
        self,
        data_directory: str = "data",
        output_directory: str = "output",
        config_file: Optional[str] = None
    ):
        """
        Initialize the Analytical Engine.
        
        Args:
            data_directory: Directory where input data is stored
            output_directory: Directory where analysis results will be saved
            config_file: Optional path to configuration file
        """
        self.data_directory = data_directory
        self.raw_data_dir = os.path.join(data_directory, "raw")
        self.processed_data_dir = os.path.join(data_directory, "processed")
        self.output_directory = output_directory
        
        # Create directories if they don't exist
        for directory in [self.data_directory, self.raw_data_dir, self.processed_data_dir, self.output_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
        
        # Default configuration
        self.config = {
            "anomaly_detection": {
                "contamination": 0.05,
                "random_state": 42
            },
            "clustering": {
                "n_clusters": 5,
                "random_state": 42
            },
            "dimension_reduction": {
                "n_components": 2
            },
            "prediction": {
                "test_size": 0.2,
                "random_state": 42
            }
        }
        
        # Load configuration if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    self.config.update(custom_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration file: {e}")
        
        # Initialize data storage
        self.datasets = {}
        self.analysis_results = {}
        
        logger.info("Analytical Engine initialized")
    
    def load_data(self, source_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from files in the data directory.
        
        Args:
            source_name: Optional filter to load data only from a specific source
            
        Returns:
            Dictionary of loaded dataframes
        """
        loaded_data = {}
        
        try:
            # Read from raw data directory
            for file_name in os.listdir(self.raw_data_dir):
                if not file_name.endswith('.json'):
                    continue
                
                # Extract source name from filename (format: sourcename_timestamp.json)
                file_source_name = file_name.split('_')[0]
                
                # Skip if source_name is specified and doesn't match
                if source_name is not None and file_source_name != source_name:
                    continue
                
                file_path = os.path.join(self.raw_data_dir, file_name)
                
                try:
                    # Load JSON data
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Convert to DataFrame
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        # Try to handle nested dictionaries or specialized formats
                        if "data" in data and isinstance(data["data"], list):
                            df = pd.DataFrame(data["data"])
                        elif "records" in data and isinstance(data["records"], list):
                            df = pd.DataFrame(data["records"])
                        else:
                            # Fallback: convert top-level dict to a single-row dataframe
                            df = pd.DataFrame([data])
                    else:
                        logger.warning(f"Unsupported data format in {file_name}, skipping")
                        continue
                    
                    # Store in datasets dictionary with source_name as key
                    if file_source_name not in loaded_data:
                        loaded_data[file_source_name] = df
                    else:
                        # Append to existing data from the same source
                        loaded_data[file_source_name] = pd.concat([loaded_data[file_source_name], df])
                    
                    logger.info(f"Loaded data from {file_name}, shape: {df.shape}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {e}")
            
            # Store in class instance
            self.datasets.update(loaded_data)
            
            logger.info(f"Successfully loaded data from {len(loaded_data)} source(s)")
            return loaded_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}
    
    def preprocess_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """
        Preprocess the data for a specific source to handle missing values, 
        outliers, and prepare it for analysis.
        
        Args:
            source_name: The name of the data source to preprocess
            
        Returns:
            Preprocessed DataFrame or None if processing failed
        """
        if source_name not in self.datasets:
            logger.error(f"Source '{source_name}' not found in loaded datasets")
            return None
        
        df = self.datasets[source_name].copy()
        logger.info(f"Preprocessing data for {source_name}, initial shape: {df.shape}")
        
        try:
            # Drop duplicate rows
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            if len(df) < initial_rows:
                logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
            
            # Handle missing values
            for column in df.columns:
                missing_count = df[column].isna().sum()
                if missing_count > 0:
                    missing_pct = missing_count / len(df)
                    logger.info(f"Column '{column}' has {missing_count} missing values ({missing_pct:.2%})")
                    
                    # Strategy depends on missing percentage and data type
                    if missing_pct > 0.5:
                        # More than 50% missing, consider dropping the column
                        logger.warning(f"Dropping column '{column}' due to high missing rate ({missing_pct:.2%})")
                        df.drop(columns=[column], inplace=True)
                    else:
                        # Less than 50% missing, impute values
                        col_dtype = df[column].dtype
                        if np.issubdtype(col_dtype, np.number):
                            # For numeric columns, fill with median
                            median_val = df[column].median()
                            df[column].fillna(median_val, inplace=True)
                            logger.info(f"Filled missing values in '{column}' with median: {median_val}")
                        elif pd.api.types.is_datetime64_any_dtype(df[column]):
                            # For datetime, use forward fill
                            df[column].fillna(method='ffill', inplace=True)
                            # If still has NAs, use backward fill
                            df[column].fillna(method='bfill', inplace=True)
                            logger.info(f"Filled missing datetime values in '{column}' using forward/backward fill")
                        else:
                            # For categorical/text, fill with most common value
                            most_common = df[column].mode()[0]
                            df[column].fillna(most_common, inplace=True)
                            logger.info(f"Filled missing values in '{column}' with mode: {most_common}")
            
            # Convert date strings to datetime objects
            for column in df.columns:
                if df[column].dtype == 'object':
                    try:
                        # Try to convert to datetime
                        df[column] = pd.to_datetime(df[column], errors='ignore')
                    except:
                        pass
            
            # Handle outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for column in numeric_cols:
                # Use IQR method to detect outliers
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in '{column}'")
                    
                    # Create a new column to mark outliers
                    outlier_col = f"{column}_is_outlier"
                    df[outlier_col] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
                    
                    # Cap outliers to bounds rather than removing them
                    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped outliers in '{column}' to range [{lower_bound}, {upper_bound}]")
            
            # Create new features that might be useful for analysis
            # 1. Date-based features from datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for column in datetime_cols:
                df[f"{column}_year"] = df[column].dt.year
                df[f"{column}_month"] = df[column].dt.month
                df[f"{column}_day"] = df[column].dt.day
                df[f"{column}_dayofweek"] = df[column].dt.dayofweek
                
                # Check if it's a timestamp with time components
                if (df[column].dt.hour != 0).any():
                    df[f"{column}_hour"] = df[column].dt.hour
                
                logger.info(f"Created date-based features from '{column}'")
            
            # Save preprocessed data
            output_path = os.path.join(self.processed_data_dir, f"{source_name}_preprocessed.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved preprocessed data to {output_path}, final shape: {df.shape}")
            
            # Update the dataset in memory
            self.datasets[f"{source_name}_preprocessed"] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data for {source_name}: {e}")
            return None
    
    def detect_anomalies(self, source_name: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect anomalies in data using Isolation Forest.
        
        Args:
            source_name: Name of the preprocessed data source
            columns: Optional list of columns to use for anomaly detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        processed_source = f"{source_name}_preprocessed"
        if processed_source not in self.datasets:
            logger.warning(f"Preprocessed data