import ast
import os
import sys
import subprocess
import time
from typing import Tuple, List, Dict, Any

class UnsafeCodeError(Exception):
    pass

class SafetyAnalyzer(ast.NodeVisitor):
    def __init__(self, allowed_tools: List[str]):
        self.allowed_tools = allowed_tools
        self.errors = []
        
        # Define forbidden modules and calls
        self.forbidden_imports = {
            'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib', 'requests', 
            'http', 'ctypes', 'builtins', 'importlib', 'pty', 'platform'
        }
        self.forbidden_calls = {
            'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 
            'getattr', 'setattr', 'delattr', 'open'
        }

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name.split('.')[0]
            if name in self.forbidden_imports:
                self.errors.append(f"Import of forbidden module '{name}' is not allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            name = node.module.split('.')[0]
            if name in self.forbidden_imports:
                self.errors.append(f"Import from forbidden module '{name}' is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.forbidden_calls:
                self.errors.append(f"Call to unsafe built-in function '{func_name}' is not allowed. Use injected tools instead.")
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name.startswith('__') and attr_name.endswith('__'):
                self.errors.append(f"Access to magic method attribute '{attr_name}' is forbidden.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        attr_name = node.attr
        if attr_name in ['__subclasses__', '__globals__', '__code__', '__builtins__']:
            self.errors.append(f"Access to sensitive attribute '{attr_name}' is forbidden.")
        self.generic_visit(node)


def verify_code_safety(code: str, allowed_tools: List[str]) -> Tuple[bool, List[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e.msg} on line {e.lineno}"]
    
    analyzer = SafetyAnalyzer(allowed_tools)
    analyzer.visit(tree)
    if analyzer.errors:
        return False, analyzer.errors
    return True, []


# --- OS / POSIX Resource Limits Setup ---
try:
    import resource
    def set_resource_limits():
        # Limit CPU time to 5 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        # Limit Maximum File Write Size to 10MB
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
except ImportError:
    # Fallback for systems that don't support resource (e.g. Windows)
    def set_resource_limits():
        pass


def get_process_metrics(pid: int) -> Tuple[float, float]:
    """
    Fetches real-time process CPU% and estimated RAM (MB) using system ps command.
    """
    try:
        res = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu,%mem"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        lines = res.stdout.strip().split("\n")
        if len(lines) > 1:
            parts = lines[1].split()
            cpu = float(parts[0])
            mem = float(parts[1])
            # Estimate RAM in MB (assuming 8GB RAM baseline)
            ram_mb = mem * 81.92
            return cpu, ram_mb
    except Exception:
        pass
    return 0.0, 0.0


def run_code_in_sandbox(
    code: str, 
    sandbox_dir: str, 
    timeout_seconds: int = 5, 
    python_executable: str = sys.executable
) -> Tuple[int, str, str]:
    """
    Runs code inside the sandbox. Uses real-time resource polling
    integrated with PyTorch Anomaly Autoencoder and POSIX resource limits.
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    temp_file = os.path.join(sandbox_dir, "temp_sandbox_run.py")
    
    injected_helpers = f"""
import os
import glob

def read_file(path):
    clean_path = os.path.normpath(os.path.join("{sandbox_dir}", path))
    if not clean_path.startswith(os.path.normpath("{sandbox_dir}")):
        raise PermissionError("Access denied: path is outside the workspace sandbox.")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"File not found: {{path}}")
    with open(clean_path, 'r') as f:
        return f.read()

def write_file(path, content):
    clean_path = os.path.normpath(os.path.join("{sandbox_dir}", path))
    if not clean_path.startswith(os.path.normpath("{sandbox_dir}")):
        raise PermissionError("Access denied: path is outside the workspace sandbox.")
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    with open(clean_path, 'w') as f:
        f.write(content)

def list_files(pattern="*"):
    search_path = os.path.join("{sandbox_dir}", pattern)
    files = glob.glob(search_path)
    return [os.path.relpath(f, "{sandbox_dir}") for f in files]

# --- AGENT CODE START ---
"""
    full_code = injected_helpers + code
    with open(temp_file, 'w') as f:
        f.write(full_code)
        
    try:
        from lume.anomaly_detector import PyTorchAnomalyDetector
        detector = PyTorchAnomalyDetector()
    except Exception as e:
        print(f"⚠️ Anomaly detector import error: {e}. Running without autoencoder.")
        detector = None

    # Spawn process asynchronously using the target virtual environment Python
    process = subprocess.Popen(
        [python_executable, temp_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=set_resource_limits
    )
    
    start_time = time.time()
    is_killed = False
    kill_reason = ""
    
    # Real-time resource monitoring loop
    while process.poll() is None:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            process.terminate()
            is_killed = True
            kill_reason = f"Execution timed out after {timeout_seconds} seconds."
            break
            
        if detector:
            cpu, ram = get_process_metrics(process.pid)
            try:
                # Count files written inside sandbox folder
                file_count = len(os.listdir(sandbox_dir))
            except OSError:
                file_count = 0
                
            is_anomalous, score = detector.is_anomalous(cpu, ram, file_count)
            if is_anomalous:
                print(f"🚨 PYTORCH AUTOENCODER WARNING: Suspicious resource activity detected (MSE Loss: {score:.4f})")
                print("🚨 Terminating process to protect host system.")
                process.terminate()
                is_killed = True
                kill_reason = f"Security Violation: PyTorch Anomaly Detector flagged resource footprint (Reconstruction Loss: {score:.4f})."
                break
                
        time.sleep(0.2)
        
    stdout, stderr = process.communicate()
    
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except OSError:
            pass
            
    if is_killed:
        return -1, "", kill_reason
    return process.returncode, stdout, stderr
