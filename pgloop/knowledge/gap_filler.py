"""
Gap Filler Module

Uses machine learning to estimate missing LCI/TEA parameters based on
similar processes and statistical inference.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class PredictionResult:
    """Result of gap filling prediction."""
    
    parameter: str
    predicted_value: float
    uncertainty_low: float
    uncertainty_high: float
    confidence: float
    method: str
    similar_sources: List[Dict]


class GapFiller:
    """
    Machine learning-based gap filler for missing LCI/TEA data.
    
    Methods:
    - Similarity matching: Find similar processes and use their values
    - Regression prediction: Train on known data to predict missing values
    - Statistical inference: Use distributions from similar processes
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the gap filler.
        
        Args:
            data_path: Path to training data / reference database
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
        
        self.data_path = data_path or Path("./data/processed")
        self.scaler = StandardScaler()
        self.reference_data = []
        self.models = {}
        
        self._load_reference_data()
    
    def _load_reference_data(self) -> None:
        """Load reference data for similarity matching."""
        ref_file = self.data_path / "reference_lci.json"
        
        if ref_file.exists():
            with open(ref_file, "r", encoding="utf-8") as f:
                self.reference_data = json.load(f)
    
    def add_reference_data(
        self,
        process_name: str,
        parameters: Dict[str, float],
        metadata: Dict = None
    ) -> None:
        """
        Add a reference process to the database.
        
        Args:
            process_name: Name of the process
            parameters: Parameter values
            metadata: Additional metadata (country, year, source)
        """
        entry = {
            "name": process_name,
            "parameters": parameters,
            "metadata": metadata or {}
        }
        self.reference_data.append(entry)
    
    def find_similar_processes(
        self,
        known_parameters: Dict[str, float],
        n_neighbors: int = 5
    ) -> List[Dict]:
        """
        Find similar processes based on known parameters.
        
        Args:
            known_parameters: Dict of known parameter values
            n_neighbors: Number of similar processes to return
            
        Returns:
            List of similar processes with similarity scores
        """
        if not self.reference_data:
            return []
        
        # Get common parameters
        param_names = list(known_parameters.keys())
        
        # Build feature matrix from reference data
        ref_vectors = []
        valid_refs = []
        
        for ref in self.reference_data:
            ref_params = ref["parameters"]
            if all(p in ref_params for p in param_names):
                vector = [ref_params[p] for p in param_names]
                ref_vectors.append(vector)
                valid_refs.append(ref)
        
        if not ref_vectors:
            return []
        
        # Fit nearest neighbors
        X = np.array(ref_vectors)
        query = np.array([[known_parameters[p] for p in param_names]])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        query_scaled = self.scaler.transform(query)
        
        # Find neighbors
        n_neighbors = min(n_neighbors, len(valid_refs))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(X_scaled)
        
        distances, indices = nn.kneighbors(query_scaled)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
            results.append({
                "process": valid_refs[idx],
                "similarity": similarity,
                "distance": dist
            })
        
        return results
    
    def predict_by_similarity(
        self,
        known_parameters: Dict[str, float],
        target_parameter: str,
        n_neighbors: int = 5
    ) -> PredictionResult:
        """
        Predict missing parameter using similarity matching.
        
        Args:
            known_parameters: Known parameter values
            target_parameter: Parameter to predict
            n_neighbors: Number of neighbors to use
            
        Returns:
            PredictionResult with prediction and uncertainty
        """
        similar = self.find_similar_processes(known_parameters, n_neighbors)
        
        if not similar:
            return PredictionResult(
                parameter=target_parameter,
                predicted_value=np.nan,
                uncertainty_low=np.nan,
                uncertainty_high=np.nan,
                confidence=0.0,
                method="similarity",
                similar_sources=[]
            )
        
        # Get target values from similar processes
        values = []
        weights = []
        sources = []
        
        for match in similar:
            proc = match["process"]
            if target_parameter in proc["parameters"]:
                values.append(proc["parameters"][target_parameter])
                weights.append(match["similarity"])
                sources.append({
                    "name": proc["name"],
                    "value": proc["parameters"][target_parameter],
                    "similarity": match["similarity"],
                    **proc.get("metadata", {})
                })
        
        if not values:
            return PredictionResult(
                parameter=target_parameter,
                predicted_value=np.nan,
                uncertainty_low=np.nan,
                uncertainty_high=np.nan,
                confidence=0.0,
                method="similarity",
                similar_sources=[]
            )
        
        # Weighted average
        values = np.array(values)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        predicted = np.average(values, weights=weights)
        
        # Uncertainty from weighted std
        variance = np.average((values - predicted) ** 2, weights=weights)
        std = np.sqrt(variance)
        
        return PredictionResult(
            parameter=target_parameter,
            predicted_value=predicted,
            uncertainty_low=predicted - 2 * std,
            uncertainty_high=predicted + 2 * std,
            confidence=np.mean(weights) * 2,  # Higher similarity = higher confidence
            method="similarity",
            similar_sources=sources
        )
    
    def train_regression_model(
        self,
        target_parameter: str,
        feature_parameters: List[str]
    ) -> float:
        """
        Train a regression model for a specific parameter.
        
        Args:
            target_parameter: Parameter to predict
            feature_parameters: Parameters to use as features
            
        Returns:
            Model R² score
        """
        # Build training data
        X = []
        y = []
        
        for ref in self.reference_data:
            params = ref["parameters"]
            if target_parameter in params and all(f in params for f in feature_parameters):
                X.append([params[f] for f in feature_parameters])
                y.append(params[target_parameter])
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data for training: {len(X)} samples")
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Store model
        self.models[target_parameter] = {
            "model": model,
            "features": feature_parameters,
            "scaler": StandardScaler().fit(X)
        }
        
        return model.score(X, y)
    
    def predict_by_regression(
        self,
        known_parameters: Dict[str, float],
        target_parameter: str
    ) -> PredictionResult:
        """
        Predict using trained regression model.
        
        Args:
            known_parameters: Known parameter values
            target_parameter: Parameter to predict
            
        Returns:
            PredictionResult with prediction
        """
        if target_parameter not in self.models:
            raise ValueError(f"No model trained for {target_parameter}")
        
        model_info = self.models[target_parameter]
        features = model_info["features"]
        
        if not all(f in known_parameters for f in features):
            missing = [f for f in features if f not in known_parameters]
            raise ValueError(f"Missing required features: {missing}")
        
        X = np.array([[known_parameters[f] for f in features]])
        
        # Get predictions from all trees for uncertainty
        model = model_info["model"]
        predictions = np.array([tree.predict(X)[0] for tree in model.estimators_])
        
        predicted = predictions.mean()
        std = predictions.std()
        
        return PredictionResult(
            parameter=target_parameter,
            predicted_value=predicted,
            uncertainty_low=predicted - 2 * std,
            uncertainty_high=predicted + 2 * std,
            confidence=0.8,  # Based on model performance
            method="regression",
            similar_sources=[]
        )
    
    def fill_gaps(
        self,
        partial_data: Dict[str, float],
        target_parameters: List[str],
        method: str = "auto"
    ) -> Dict[str, PredictionResult]:
        """
        Fill multiple missing parameters.
        
        Args:
            partial_data: Known parameter values
            target_parameters: Parameters to predict
            method: Prediction method (similarity, regression, auto)
            
        Returns:
            Dict mapping parameter name to prediction result
        """
        results = {}
        
        for param in target_parameters:
            if method == "similarity" or method == "auto":
                result = self.predict_by_similarity(partial_data, param)
            elif method == "regression":
                try:
                    result = self.predict_by_regression(partial_data, param)
                except ValueError:
                    result = self.predict_by_similarity(partial_data, param)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[param] = result
        
        return results
    
    def save_reference_data(self) -> None:
        """Save reference data to disk."""
        ref_file = self.data_path / "reference_lci.json"
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        with open(ref_file, "w", encoding="utf-8") as f:
            json.dump(self.reference_data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    filler = GapFiller()
    
    # Add some reference data
    filler.add_reference_data(
        "PG_Cement_China",
        {"energy_mj": 150, "co2_kg": 25, "water_m3": 2.5, "efficiency": 0.95},
        {"country": "China", "year": 2023}
    )
    filler.add_reference_data(
        "PG_Cement_Morocco",
        {"energy_mj": 180, "co2_kg": 30, "water_m3": 3.0, "efficiency": 0.92},
        {"country": "Morocco", "year": 2022}
    )
    
    # Predict missing parameter
    known = {"energy_mj": 160, "co2_kg": 27}
    result = filler.predict_by_similarity(known, "water_m3")
    
    print(f"Predicted water_m3: {result.predicted_value:.2f}")
    print(f"Uncertainty: [{result.uncertainty_low:.2f}, {result.uncertainty_high:.2f}]")
