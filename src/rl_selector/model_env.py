"""
DataPilot AI Pro - Model Selection Environment
===============================================
Gymnasium environment for training the PPO model selector.

Environment Design:
- State: 30+ meta-features from dataset
- Action: Select one model from pool
- Reward: Model performance (F1 score) on test set
- Episode: Complete when all models evaluated or budget exhausted
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


if HAS_GYM:
    class ModelSelectionEnv(gym.Env):
        """
        Gymnasium environment for model selection RL training.
        
        The agent learns to select optimal ML models by observing
        dataset meta-features and receiving rewards based on
        model performance.
        """
        
        metadata = {"render_modes": ["human"]}
        
        def __init__(
            self,
            n_meta_features: int = 30,
            n_models: int = 5,
            max_selections: int = 3,
        ):
            """
            Initialize the model selection environment.
            
            Args:
                n_meta_features: Dimension of meta-feature vector
                n_models: Number of available models
                max_selections: Maximum models to select per episode
            """
            super().__init__()
            
            self.n_meta_features = n_meta_features
            self.n_models = n_models
            self.max_selections = max_selections
            
            # State: meta-features + selection mask
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_meta_features + n_models,),
                dtype=np.float32,
            )
            
            # Action: select a model
            self.action_space = spaces.Discrete(n_models)
            
            # Environment state
            self.meta_features: Optional[np.ndarray] = None
            self.selected_models: list = []
            self.selection_mask: np.ndarray = np.zeros(n_models)
            self.model_performances: Dict[int, float] = {}
            self.steps: int = 0
        
        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None,
        ) -> Tuple[np.ndarray, Dict]:
            """Reset environment with new dataset meta-features."""
            super().reset(seed=seed)
            
            # Generate random meta-features (in training, use real data)
            if options and "meta_features" in options:
                self.meta_features = options["meta_features"]
            else:
                self.meta_features = self._generate_random_meta_features()
            
            # Reset selection state
            self.selected_models = []
            self.selection_mask = np.zeros(self.n_models)
            self.model_performances = self._simulate_model_performances()
            self.steps = 0
            
            obs = self._get_observation()
            info = {"n_models": self.n_models, "max_selections": self.max_selections}
            
            return obs, info
        
        def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            Execute model selection action.
            
            Args:
                action: Index of model to select
                
            Returns:
                observation, reward, terminated, truncated, info
            """
            terminated = False
            truncated = False
            reward = 0.0
            
            # Check if model already selected
            if self.selection_mask[action] == 1:
                reward = -0.1  # Penalty for selecting same model
            else:
                # Select the model
                self.selected_models.append(action)
                self.selection_mask[action] = 1
                
                # Reward is model performance
                reward = self.model_performances.get(action, 0.0)
            
            self.steps += 1
            
            # Episode termination
            if len(self.selected_models) >= self.max_selections:
                terminated = True
                # Bonus for selecting complementary models
                if len(set(self.selected_models)) == self.max_selections:
                    reward += 0.1
            
            obs = self._get_observation()
            info = {
                "selected_models": self.selected_models,
                "total_reward": reward,
            }
            
            return obs, reward, terminated, truncated, info
        
        def _get_observation(self) -> np.ndarray:
            """Construct observation from meta-features and selection mask."""
            return np.concatenate([
                self.meta_features,
                self.selection_mask,
            ]).astype(np.float32)
        
        def _generate_random_meta_features(self) -> np.ndarray:
            """Generate random meta-features for training."""
            return np.random.randn(self.n_meta_features).astype(np.float32)
        
        def _simulate_model_performances(self) -> Dict[int, float]:
            """Simulate model performances (in training, use real evaluations)."""
            # Base performance influenced by meta-features
            base_perf = 0.5 + 0.3 * np.random.random(self.n_models)
            
            # Add some correlation with meta-features
            if self.meta_features is not None:
                meta_effect = 0.1 * np.tanh(self.meta_features[:self.n_models].sum())
                base_perf += meta_effect
            
            return {i: float(p) for i, p in enumerate(base_perf)}
        
        def render(self) -> None:
            """Render environment state."""
            print(f"Step {self.steps}")
            print(f"Selected: {self.selected_models}")
            print(f"Mask: {self.selection_mask}")

else:
    # Fallback when Gymnasium is not installed
    class ModelSelectionEnv:
        """Placeholder when Gymnasium is not available."""
        
        def __init__(self, *args, **kwargs):
            logger.warning("Gymnasium not installed. ModelSelectionEnv unavailable.")
            raise ImportError("gymnasium is required for ModelSelectionEnv")
