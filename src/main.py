import os
import sys
import warnings
import argparse
import importlib.util
import torch
from omegaconf import OmegaConf

from trainer_manager import TrainingManager

# Setup PyTorch backend
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
# Only filter specific warnings instead of all
warnings.filterwarnings('ignore', category=UserWarning)

def load_config(config_path: str):
    """
    Load configuration file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    try:
        spec = importlib.util.spec_from_file_location("_C", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["_C"] = config_module
        spec.loader.exec_module(config_module)
        cfg = config_module._C
        
        # Convert to OmegaConf configuration object
        cfg = OmegaConf.create(cfg.__dict__)
        return cfg
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Hierarchical Multi-Instance Learning Training Script"
    )
    parser.add_argument(
        "--config",
        default="config",
        help="config file path",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)


        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus

        # FC-MIL 关键配置校验
        print(f"[FC-MIL] Initializing with alpha_cr: {getattr(cfg, 'alpha_cr', 0.1)}")
        print(f"[FC-MIL] Drop-TopK ratio set to: {getattr(cfg, 'topk_ratio', 0.03)}")

        trainer = TrainingManager(cfg)


        training_folds = [int(f) for f in cfg.folds.split(',')]

        if len(training_folds) > 1:
            print(f"[logs] Conducting {len(training_folds)}-fold cross-validation")
            results = trainer.run_cross_validation(training_folds)
        else:
            print(f"[logs] Starting single fold training for FC-MIL")
            results = trainer.train()  

        print(f"Final Results Summary: {results}")

    except Exception as e:
        print(f"Error during FC-MIL execution: {e}")
        raise

if __name__ == '__main__':
    main()