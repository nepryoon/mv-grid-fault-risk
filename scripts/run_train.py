from mvgrid.config import load_settings
from mvgrid.train import train_model


if __name__ == "__main__":
    cfg_path = "configs/config.yaml"
    settings = load_settings(cfg_path)
    train_model(settings, config_path=cfg_path)
