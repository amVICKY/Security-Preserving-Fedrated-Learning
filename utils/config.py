import os
import yaml

# Resolve the config path relative to the repo root (this file lives in utils/),
# so it works on Windows and inside a Linux container alike. CONFIG_PATH env var
# overrides it when needed (e.g. a mounted config in Docker/k8s).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
configuration_path = os.environ.get(
    "CONFIG_PATH",
    os.path.join(_REPO_ROOT, "configs", "config.yaml"),
)

def load_config(config_path=configuration_path):
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)

    return config

traing_dict = load_config()
