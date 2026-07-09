import os
import yaml
from typing import Dict, Any, List

class AgentConfig:
    def __init__(self, name: str, models: List[str], system_prompt: str, allowed_tools: List[str], sandbox_config: Dict[str, Any]):
        self.name = name
        self.models = models
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.sandbox_config = sandbox_config

def load_agent_config(filepath: str) -> AgentConfig:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    name = data.get("name", "anonymous_agent")
    models = data.get("models", ["llama3:latest"])
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    allowed_tools = data.get("allowed_tools", [])
    sandbox_config = data.get("sandbox", {
        "timeout_seconds": 5,
        "block_unsafe_imports": True
    })
    
    return AgentConfig(
        name=name,
        models=models,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        sandbox_config=sandbox_config
    )
