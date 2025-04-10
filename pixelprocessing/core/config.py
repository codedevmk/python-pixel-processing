import os
import json
import logging
from typing import Dict, Any, Optional, Union

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class Config:
    """
    Configuration management for pixel processing.
    
    This class handles loading, saving, and validating configuration settings
    for the pixel processing library. It supports loading from JSON files,
    dictionaries, or environment variables.
    """
    
    def __init__(self, 
                 config: Optional[Union[str, Dict[str, Any]]] = None, 
                 base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config: Configuration source (file path or dictionary)
            base_config: Base configuration to merge with loaded config
        """
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize default configuration
        self._config = {
            # General settings
            'process_transparent': False,  # Whether to process transparent images
            
            # File settings
            'supported_extensions': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
            'image_quality': 95,  # Default quality for saving
            'image_format': 'PNG',  # Default format for saving
            
            # Color settings
            'color_distance_threshold': 30,
            'num_dominant_colors': 8,
            
            # Resize settings
            'resize_method': 'pixel-perfect',
            
            # Dither settings
            'dither_type': 'none',  # Default to no dithering
            'dither_strength': 1.0,  # Default dithering strength
            
            # Mask settings
            'generate_mask': False,  # Whether to generate mask images
            'mask_type': 'edge',     # Default mask type
            'mask_threshold': 30,    # Default threshold
            'invert_mask': False,    # Whether to invert masks
            'blur_mask': False,      # Whether to blur masks
            
            # Background removal settings
            'remove_bg': False,          # Whether to remove background
            'bg_method': 'color',        # Background removal method
            'bg_threshold': 30,          # Background detection threshold
            'bg_color': None,            # Background color (if manually specified)
            'feather_edges': False,      # Whether to smooth edges
        }
        
        # Apply base config if provided
        if base_config:
            self._deep_merge(base_config)
        
        # Load configuration if provided
        if config:
            self.load(config)
    
    def load(self, config: Union[str, Dict[str, Any]]) -> None:
        """
        Load configuration from various sources.
        
        Args:
            config: Configuration source (file path or dictionary)
        
        Raises:
            ConfigurationError: If loading fails
        """
        try:
            # If string, treat as file path
            if isinstance(config, str):
                # Expand user and environment variables
                config_path = os.path.expanduser(os.path.expandvars(config))
                
                # Validate file exists
                if not os.path.exists(config_path):
                    raise ConfigurationError(f"Configuration file not found: {config_path}")
                
                # Load JSON configuration
                with open(config_path, 'r') as config_file:
                    loaded_config = json.load(config_file)
            
            # If dictionary, use directly
            elif isinstance(config, dict):
                loaded_config = config
            
            else:
                raise ConfigurationError("Invalid configuration type. Must be path or dictionary.")
            
            # Merge loaded configuration
            self._deep_merge(loaded_config)
            
            self.logger.info("Configuration loaded successfully")
        
        except json.JSONDecodeError:
            raise ConfigurationError(f"Invalid JSON in configuration file: {config}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _deep_merge(self, new_config: Dict[str, Any]) -> None:
        """
        Perform a deep merge of configurations.
        
        Args:
            new_config: Configuration to merge
        """
        def merge_dict(original, update):
            """Recursive dictionary merge."""
            for key, value in update.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    # Recursively merge nested dictionaries
                    original[key] = merge_dict(original[key], value)
                else:
                    # Replace or add key-value pair
                    original[key] = value
            return original
        
        self._config = merge_dict(self._config, new_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value with optional default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self._config[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to configuration values.
        
        Args:
            name: Configuration key
        
        Returns:
            Configuration value
        
        Raises:
            AttributeError: If the key is not found
        """
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Allow attribute-style setting of configuration values.
        
        Args:
            name: Configuration key
            value: Value to set
        """
        if name.startswith('_') or name == 'logger':
            # Private attributes
            super().__setattr__(name, value)
        else:
            # Configuration keys
            self._config[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return dict(self._config)
    
    def save(self, path: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            path: File path to save configuration
        
        Raises:
            ConfigurationError: If saving fails
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save configuration
            with open(path, 'w') as config_file:
                json.dump(self._config, config_file, indent=2)
            
            self.logger.info(f"Configuration saved to {path}")
        
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
