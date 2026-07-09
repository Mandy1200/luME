import uuid
import os
from typing import Dict, Any, List, TypedDict, Literal
from langgraph.graph import StateGraph, END

# Import local luME modules
from lume.pytorch_classifier import PyTorchIntentClassifier
from lume.rlsf import PyTorchRLSFRouter
from lume.telemetry import log_telemetry_event
from lume.semantic_fs import semantic_search
from lume.debate import run_debate, query_model, extract_python_code
from lume.sandbox import verify_code_safety
from lume.env_manager import setup_sandbox_virtualenv
from lume.sandbox import run_code_in_sandbox

# Define State Schema
class AgentState(TypedDict):
    task: str
    session_id: str
    intent: str
    intent_logits: List[float]  # Simulated logits based on intent prediction
    allowed_tools: List[str]
    tool_mask: List[bool]
    saved_probs: List[Any]
    code: str
    output: str
    run_count: int
    success: bool
    models: List[str]
    system_prompt: str
    sandbox_dir: str
    timeout_seconds: int

# Initialize local PyTorch classifiers & routers
try:
    classifier = PyTorchIntentClassifier()
    rlsf_router = PyTorchRLSFRouter()
except Exception as e:
    print(f"⚠️ PyTorch initialization deferred: {e}")
    classifier = None
    rlsf_router = None


# --- GRAPH NODES ---

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    task = state["task"]
    log_telemetry_event(session_id, "intent_classifier", "START", f"Classifying query: {task}")
    
    if classifier:
        intent = classifier.predict(task)
        # Convert prediction to simulated logits for RLSF input
        intent_map = {"DATA_SCIENCE": [1.0, 0.0, 0.0], "FILE_OPERATION": [0.0, 1.0, 0.0], "GENERAL_CHAT": [0.0, 0.0, 1.0]}
        logits = intent_map.get(intent, [0.0, 0.0, 1.0])
    else:
        intent = "GENERAL_CHAT"
        logits = [0.0, 0.0, 1.0]
        
    log_telemetry_event(session_id, "intent_classifier", "COMPLETE", f"Intent: {intent}", {"logits": logits})
    return {"intent": intent, "intent_logits": logits}


def rlsf_router_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    logits = state.get("intent_logits", [0.0, 0.0, 1.0])
    log_telemetry_event(session_id, "rlsf_router", "START", "Selecting tool permissions...")
    
    tools = ["read_file", "write_file", "list_files"]
    if rlsf_router:
        mask, saved_probs = rlsf_router.select_tools(logits)
    else:
        mask = [True, True, True]
        saved_probs = []
        
    allowed = [tools[i] for i, enabled in enumerate(mask) if enabled]
    
    log_telemetry_event(
        session_id, 
        "rlsf_router", 
        "COMPLETE", 
        f"Allowed tools: {allowed}", 
        {"mask": mask}
    )
    return {"allowed_tools": allowed, "tool_mask": mask, "saved_probs": saved_probs}


def context_injection_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    task = state["task"]
    intent = state["intent"]
    
    # Only run search if task involves files or data science
    if intent in ["DATA_SCIENCE", "FILE_OPERATION"]:
        log_telemetry_event(session_id, "rag_context", "START", "Querying semantic memory index...")
        hits = semantic_search(task, top_k=2)
        if hits:
            context = "\n\n=== RELEVANT CONTEXT FILES ===\n"
            for hit in hits:
                context += f"File: {hit['filepath']} (Relevance: {hit['similarity']:.3f})\nContent:\n{hit['content']}\n---\n"
            
            updated_task = task + context
            log_telemetry_event(session_id, "rag_context", "COMPLETE", f"Injected {len(hits)} matching context blocks.")
            return {"task": updated_task}
            
    return {}


def debate_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    log_telemetry_event(session_id, "debate_arena", "START", "Running Multi-LLM Debate Arena...")
    
    # We append tool guidelines so the models know they cannot use open()
    tools_str = ", ".join(state["allowed_tools"])
    guidelines = f"\nAllowed tools you can call: {tools_str}. Do not use standard open() or import os. Use read_file, write_file, or list_files."
    prompt = state["task"] + guidelines
    
    code = run_debate(state["models"], state["system_prompt"], prompt)
    log_telemetry_event(session_id, "debate_arena", "COMPLETE", f"Synthesized code proposal ({len(code)} chars)")
    return {"code": code}


def ast_sandbox_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    code = state["code"]
    sandbox_dir = state["sandbox_dir"]
    allowed = state["allowed_tools"]
    timeout = state["timeout_seconds"]
    run_count = state["run_count"] + 1
    
    log_telemetry_event(session_id, "ast_sandbox", "START", f"Attempt {run_count}: Verifying safety & running script...")
    
    # 1. AST Safety Check
    is_safe, safety_errors = verify_code_safety(code, allowed)
    if not is_safe:
        error_msg = f"Safety Violations: " + ", ".join(safety_errors)
        log_telemetry_event(session_id, "ast_sandbox", "VIOLATION", error_msg)
        return {"output": error_msg, "success": False, "run_count": run_count}
        
    # 2. Dynamic environment setup (Stage 5)
    python_exec = setup_sandbox_virtualenv(sandbox_dir, code)
    
    # 3. Subprocess execution with POSIX limits & Autoencoder monitoring (Stage 6 & 7)
    exit_code, stdout, stderr = run_code_in_sandbox(
        code=code, 
        sandbox_dir=sandbox_dir, 
        timeout_seconds=timeout, 
        python_executable=python_exec
    )
    
    if exit_code == 0:
        log_telemetry_event(session_id, "ast_sandbox", "SUCCESS", "Code executed cleanly in sandbox.")
        return {"output": stdout, "success": True, "run_count": run_count}
    else:
        error_msg = stderr if stderr else stdout
        log_telemetry_event(session_id, "ast_sandbox", "FAILURE", f"Execution failed: {error_msg}")
        return {"output": error_msg, "success": False, "run_count": run_count}


def self_heal_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    code = state["code"]
    error_msg = state["output"]
    referee_model = state["models"][0]
    
    log_telemetry_event(session_id, "self_healing", "START", "Querying LLM to correct errors...")
    
    repair_prompt = (
        f"Your code was executed in a sandbox and failed with this error/violation:\n{error_msg}\n\n"
        "Please rewrite the Python code to remove these bugs or violations. "
        "Ensure you start directly with Python code (do not explain the bug, just output the corrected script).\n\n"
        f"Faulty Code:\n{code}\n\n"
        "Provide ONLY the final corrected Python code. Do not include explanation text."
    )
    raw_repair = query_model(referee_model, "You are a helpful software engineer debugging sandbox errors.", repair_prompt)
    healed_code = extract_python_code(raw_repair)
    
    log_telemetry_event(session_id, "self_healing", "COMPLETE", "Healed script generated.")
    return {"code": healed_code}


def rlsf_update_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    logits = state.get("intent_logits", [0.0, 0.0, 1.0])
    mask = state["tool_mask"]
    success = state["success"]
    
    reward = 1.0 if success else -1.0
    log_telemetry_event(session_id, "rlsf_update", "START", f"Updating router policy with reward: {reward}")
    
    if rlsf_router:
        rlsf_router.update_policy(logits, mask, reward)
        log_telemetry_event(session_id, "rlsf_update", "COMPLETE", "Policy updated successfully.")
    else:
        log_telemetry_event(session_id, "rlsf_update", "SKIPPED", "RLSF Router model was not loaded.")
        
    return {}


# --- TRANSITION ROUTERS ---

def route_after_execution(state: AgentState) -> Literal["self_heal", "rlsf_update", "__end__"]:
    if state["success"]:
        return "rlsf_update"
    
    # If failed but we have retries left, self-heal
    if state["run_count"] < 3:
        return "self_heal"
        
    # Out of retries, log failure and update RLSF negatively
    return "rlsf_update"


# --- BUILD STATE GRAPH ---

def build_orchestration_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("rlsf_router", rlsf_router_node)
    workflow.add_node("context_injection", context_injection_node)
    workflow.add_node("debate_arena", debate_node)
    workflow.add_node("ast_sandbox", ast_sandbox_node)
    workflow.add_node("self_heal", self_heal_node)
    workflow.add_node("rlsf_update", rlsf_update_node)
    
    # Define Edges (Transitions)
    workflow.set_entry_point("classify_intent")
    
    workflow.add_edge("classify_intent", "rlsf_router")
    workflow.add_edge("rlsf_router", "context_injection")
    workflow.add_edge("context_injection", "debate_arena")
    workflow.add_edge("debate_arena", "ast_sandbox")
    
    # Cyclic execution transition conditional edges
    workflow.add_conditional_edges(
        "ast_sandbox",
        route_after_execution,
        {
            "self_heal": "self_heal",
            "rlsf_update": "rlsf_update",
            "__end__": END
        }
    )
    
    workflow.add_edge("self_heal", "ast_sandbox")
    workflow.add_edge("rlsf_update", END)
    
    return workflow.compile()


def execute_agent_task(
    task: str,
    models: List[str],
    system_prompt: str,
    sandbox_dir: str = "sandbox_workspace",
    timeout_seconds: int = 5
) -> AgentState:
    """
    Kicks off the state graph execution pipeline for a given user task.
    """
    graph = build_orchestration_graph()
    
    initial_state = {
        "task": task,
        "session_id": str(uuid.uuid4()),
        "intent": "GENERAL_CHAT",
        "intent_logits": [0.0, 0.0, 1.0],
        "allowed_tools": [],
        "tool_mask": [],
        "saved_probs": [],
        "code": "",
        "output": "",
        "run_count": 0,
        "success": False,
        "models": models,
        "system_prompt": system_prompt,
        "sandbox_dir": sandbox_dir,
        "timeout_seconds": timeout_seconds
    }
    
    print("🕸️ Compiling and running LangGraph state workflow...")
    final_state = graph.invoke(initial_state)
    return final_state
