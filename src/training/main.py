import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG_DIR, CONFIG_NAME
import hydra
from training.train import train
from omegaconf import DictConfig


@hydra.main(version_base="1.1", config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(config: DictConfig):
    train(config)

if __name__ == "__main__":
    main()