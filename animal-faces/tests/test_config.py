import yaml
from pathlib import Path

def test_config_loads():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    print(f"Checking config.yaml at: {config_path}")
    assert config_path.exists(), "config.yaml 文件不存在"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert "models" in config
    assert "hybrid" in config["models"]
    assert "backbone" in config["models"]["hybrid"]
    assert config["models"]["hybrid"]["backbone"] in ["resnet18", "resnet50"]