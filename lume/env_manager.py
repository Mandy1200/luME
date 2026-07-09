import os
import sys
import subprocess
import ast
from typing import Set

class ImportDetector(ast.NodeVisitor):
    def __init__(self):
        self.modules = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.modules.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.modules.add(node.module.split('.')[0])
        self.generic_visit(node)


def detect_external_packages(code: str) -> Set[str]:
    """
    Parses code with AST to identify all module imports.
    Filters out standard library modules and returns a list of external packages.
    """
    # Standard Python library module list
    std_lib = {
        'math', 'os', 'sys', 'json', 'datetime', 'time', 'random', 're', 
        'collections', 'itertools', 'functools', 'urllib', 'hashlib', 'sqlite3',
        'glob', 'shutil', 'subprocess', 'ast', 'logging', 'uuid', 'csv', 'io'
    }
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
        
    detector = ImportDetector()
    detector.visit(tree)
    
    # Return imported packages that are not in Python's standard library
    external = detector.modules - std_lib
    
    # Map common imports to pip install names (e.g. sklearn -> scikit-learn)
    pip_mapping = {
        "sklearn": "scikit-learn",
        "PIL": "pillow"
    }
    return {pip_mapping.get(pkg, pkg) for pkg in external if pkg}


def setup_sandbox_virtualenv(sandbox_dir: str, code: str) -> str:
    """
    Creates a virtual environment in sandbox_dir (if missing),
    detects packages needed, installs them, and returns the path to the virtualenv python executable.
    """
    venv_dir = os.path.join(sandbox_dir, "venv")
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # 1. Create virtualenv if it does not exist
    if not os.path.exists(venv_dir):
        print(f"📦 Creating isolated virtual environment in {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        
    venv_python = os.path.join(venv_dir, "bin", "python3")
    if not os.path.exists(venv_python):
        # Handle Windows environments fallback
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        
    # 2. Detect required packages
    packages = detect_external_packages(code)
    
    # 3. Install packages if any are detected
    if packages:
        print(f"📦 Sandbox packages detected: {packages}. Installing dynamically...")
        venv_pip = os.path.join(venv_dir, "bin", "pip")
        if not os.path.exists(venv_pip):
            venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
            
        try:
            # Install in the isolated venv
            subprocess.run([venv_pip, "install", *packages], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ Dynamic environment packages installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install sandbox packages: {e}")
            
    return venv_python
