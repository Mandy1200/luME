import os
from typing import Tuple, List
from lume.sandbox import verify_code_safety, run_code_in_sandbox
from lume.debate import query_model, extract_python_code

def heal_and_execute(
    initial_code: str,
    models: List[str],
    allowed_tools: List[str],
    sandbox_dir: str,
    timeout_seconds: int = 5,
    max_retries: int = 3
) -> Tuple[bool, str, str]:
    """
    Executes code inside the sandbox. If safety checks or execution fails,
    asks the model to self-heal and retry.
    Returns (success, final_output, final_code).
    """
    current_code = initial_code
    referee_model = models[0]  # The first model listed acts as the healing engine
    
    for attempt in range(1, max_retries + 1):
        print(f"\n🛠️ Execution Attempt {attempt}/{max_retries}...")
        
        # Step 1: Run safety analyzer (AST check)
        is_safe, safety_errors = verify_code_safety(current_code, allowed_tools)
        if not is_safe:
            error_msg = "\n".join(safety_errors)
            print(f"⚠️ Safety violations found:\n{error_msg}")
            
            if attempt == max_retries:
                return False, f"Safety violation after {max_retries} retries: {error_msg}", current_code
                
            print("🔄 Prompting LLM to correct safety issues...")
            repair_prompt = (
                f"Your code violated our security rules:\n{error_msg}\n\n"
                "Please rewrite the Python code to remove these violations. "
                "Ensure you DO NOT import blocked modules (like os, sys, subprocess, requests) "
                "and DO NOT call open(). Instead use injected tools like read_file, write_file, or list_files.\n\n"
                f"Original Code:\n{current_code}\n\n"
                "Provide ONLY the final corrected Python code. Do not include explanation text."
            )
            raw_repair = query_model(referee_model, "You are a secure coding assistant.", repair_prompt)
            current_code = extract_python_code(raw_repair)
            continue
            
        # Step 2: Run code in restricted sandbox subprocess
        exit_code, stdout, stderr = run_code_in_sandbox(current_code, sandbox_dir, timeout_seconds)
        
        if exit_code == 0:
            print("✨ Code executed successfully!")
            return True, stdout, current_code
        else:
            error_msg = stderr if stderr else stdout
            print(f"❌ Execution failed with exit code {exit_code}.\nError details:\n{error_msg}")
            
            if attempt == max_retries:
                return False, f"Execution failed after {max_retries} retries: {error_msg}", current_code
                
            print("🔄 Prompting LLM to self-heal/repair the bug...")
            repair_prompt = (
                f"Your code was executed in a sandbox and failed with this error:\n{error_msg}\n\n"
                f"Here was the stdout:\n{stdout}\n\n"
                "Please fix the bugs in your code. Ensure it starts with standard imports or statements. "
                "Do not explain the bug. Just output the corrected code.\n\n"
                f"Faulty Code:\n{current_code}\n\n"
                "Provide ONLY the final corrected Python code. Do not include explanation text."
            )
            raw_repair = query_model(referee_model, "You are a helpful software engineer debugging code.", repair_prompt)
            current_code = extract_python_code(raw_repair)
            
    return False, "Max retries reached without success.", current_code
