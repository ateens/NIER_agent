import torch
import numpy as np
from typing import List, Optional
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import os
import logging

from .trep import TRep

logger = logging.getLogger(__name__)

class TRepEmbedding(EmbeddingFunction):
    """
    ChromaDB Embedding Function using T-Rep (Time-series Representation Learning).
    
    T-Rep is a self-supervised representation learning method for time series
    that learns robust temporal embeddings through multiple pretext tasks.
    """
    
    def __init__(
        self,
        weight_path: str,
        device: str = 'cpu',
        encoding_window: Optional[str] = 'full_series',
        input_dims: int = 1,
        output_dims: int = 128,
        time_embedding: Optional[str] = None,
        return_time_embeddings: bool = False
    ):
        """
        Initialize T-Rep embedding function.
        
        Args:
            weight_path: Path to the pre-trained T-Rep model weights file.
            device: Device to run the model on ('cpu' or 'cuda').
            encoding_window: Pooling strategy for temporal dimension:
                - 'full_series': Collapse time dimension to 1 (max pooling)
                - int: Pooling kernel size
                - 'multiscale': Combine representations at different time scales
                - None: No pooling, keep original time dimension
            input_dims: Number of input features/channels (default: 1 for univariate).
            output_dims: Dimension of the learned representation (default: 128).
            time_embedding: Type of time embedding to use (e.g., 'learnable', 't2v_sin').
            return_time_embeddings: Whether to include time embeddings in output.
        """
        self.device = device
        self.encoding_window = encoding_window
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.time_embedding = time_embedding
        self.return_time_embeddings = return_time_embeddings
        
        # Load the T-Rep model
        self.model = self._load_model(weight_path)
        
        logger.info(f"TRepEmbedding initialized with device={device}, "
                   f"encoding_window={encoding_window}, "
                   f"time_embedding={time_embedding}")
    
    def _load_model(self, weight_path: str) -> TRep:
        """
        Load pre-trained T-Rep model from weight file.
        
        Args:
            weight_path: Path to the model weights file.
            
        Returns:
            Loaded T-Rep model.
            
        Raises:
            FileNotFoundError: If weight file doesn't exist.
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
        logger.info(f"Loading T-Rep model from {weight_path}...")
        
        # Initialize T-Rep model
        model = TRep(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            device=self.device,
            time_embedding=self.time_embedding
        )
        
        # Load pre-trained weights
        state_dict = torch.load(weight_path, map_location=self.device)
        model.net.load_state_dict(state_dict)
        model.net.eval()
        
        logger.info("T-Rep model loaded successfully.")
        return model
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for input documents (time series).
        
        Args:
            input: List of document strings, where each string is a comma-separated
                   sequence of numeric values representing a time series.
        
        Returns:
            List of embedding vectors (flattened to 1D lists).
        """
        logger.info(f"Generating T-Rep embeddings for {len(input)} time series...")
        
        # Convert string documents to numpy arrays
        input_data = [self._process_values_string(doc) for doc in input]
        input_data = np.array(input_data, dtype=np.float32)
        
        # Generate embeddings using T-Rep
        if self.return_time_embeddings:
            embeddings, time_embeddings = self.model.encode(
                data=input_data,
                encoding_window=self.encoding_window,
                batch_size=len(input_data),
                return_time_embeddings=True
            )
            # Concatenate time series embeddings with time embeddings
            embeddings = np.concatenate([embeddings, time_embeddings], axis=-1)
        else:
            embeddings = self.model.encode(
                data=input_data,
                encoding_window=self.encoding_window,
                batch_size=len(input_data),
                return_time_embeddings=False
            )
        
        # Flatten embeddings to 1D lists
        flattened_embeddings = [embedding.flatten().tolist() for embedding in embeddings]
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return flattened_embeddings
    
    def _process_values_string(self, values_str: str) -> List[List[float]]:
        """
        Convert comma-separated value string to time series array.
        
        This method parses a string of comma-separated numeric values and
        converts it into a 2D array format expected by T-Rep:
        - Shape: (n_timestamps, n_features)
        - Missing values (999999.0) are converted to NaN
        
        Args:
            values_str: Comma-separated string of numeric values.
            
        Returns:
            2D list of floats representing the time series.
            
        Raises:
            ValueError: If string parsing fails.
        """
        try:
            values = [
                float(val) if float(val) != 999999.0 else float('nan')
                for val in values_str.split(',')
            ]
            # T-Rep expects shape (n_timestamps, n_features)
            # For univariate series, wrap each value in a list
            return [[val] for val in values]
        except ValueError as e:
            logger.error(f"Error parsing values string: {values_str}")
            raise e
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate similarity between two embeddings using Euclidean distance.
        
        Args:
            embedding1: First embedding vector (NumPy array or list).
            embedding2: Second embedding vector (NumPy array or list).
            
        Returns:
            Euclidean distance between the two embeddings.
        """
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Euclidean distance: sqrt(sum((x1 - x2)^2))
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance
    
    def save_model(self, save_path: str):
        """
        Save the current model state to a file.
        
        Args:
            save_path: Path where to save the model weights.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.net.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
