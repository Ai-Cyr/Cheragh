"""FastAPI example for cheragh v0.4."""
from cheragh.server import create_app

# Option 1: use a config file.
app = create_app(config_path="rag.yaml")

# Option 2: use a prebuilt local index instead.
# app = create_app(index_path=".cheragh_index")
