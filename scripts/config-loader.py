# scripts/config_loader.py

import os
import yaml
import logging
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir='configs'):
        self.config_dir = config_dir
        self.logger = logging.getLogger(__name__)
        self.config_files = [
            'config.yml', 'data_paths.yml', 'preprocessing.yml', 'models.yml',
            'ensemble.yml', 'logging.yml', 'output.yml', 'mlflow.yml'
        ]

    def load_config(self, filename='config.yml') -> Dict[str, Any]:
        def include_yaml(loader, node):
            file_path = os.path.join(self.config_dir, loader.construct_scalar(node))
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)

        yaml.add_constructor('!include', include_yaml, Loader=yaml.SafeLoader)

        inuse_path = os.path.join(self.config_dir, 'inuse', filename)
        if os.path.exists(inuse_path):
            with open(inuse_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Loaded configuration from {inuse_path}")
                return config

        config_path = os.path.join(self.config_dir, filename)
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config

    def save_config(self, config: Dict[str, Any], filename='config.yml'):
        inuse_path = os.path.join(self.config_dir, 'inuse', filename)
        os.makedirs(os.path.dirname(inuse_path), exist_ok=True)
        with open(inuse_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, allow_unicode=True)
        self.logger.info(f"Saved configuration to {inuse_path}")

    def backup_config(self, filename='config.yml'):
        source_path = os.path.join(self.config_dir, 'inuse', filename)
        if not os.path.exists(source_path):
            self.logger.warning(f"No configuration file found at {source_path}")
            return

        backup_dir = os.path.join(self.config_dir, 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{filename}.bak")
        
        with open(source_path, 'r', encoding='utf-8') as source, open(backup_path, 'w', encoding='utf-8') as backup:
            backup.write(source.read())
        self.logger.info(f"Backed up configuration to {backup_path}")

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        all_configs = {}
        for filename in self.config_files:
            all_configs[filename] = self.load_config(filename)
        return all_configs

    def validate_configs(self, configs: Dict[str, Dict[str, Any]]):
        # Add validation logic here
        required_keys = {
            'config.yml': ['data_paths', 'preprocessing', 'models', 'ensemble', 'logging', 'output', 'mlflow'],
            'data_paths.yml': ['raw', 'interim', 'processed'],
            'models.yml': ['advanced_transformer', 'advanced_gat', 'gradient_boosting'],
            # Add more required keys for other config files
        }
        
        for filename, required in required_keys.items():
            if filename not in configs:
                raise ValueError(f"Missing required config file: {filename}")
            for key in required:
                if key not in configs[filename]:
                    raise ValueError(f"Missing required key '{key}' in {filename}")

        self.logger.info("All configurations validated successfully")

if __name__ == "__main__":
    loader = ConfigLoader()
    all_configs = loader.load_all_configs()
    loader.validate_configs(all_configs)
    print("All configurations loaded and validated successfully")