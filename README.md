# luME: A Secure, Self-Healing Multi-Agent Cognitive OS Sandbox

**luME** (pronounced *lume*) is an advanced, research-grade Cognitive Agent Operating System. Unlike standard LLM projects that act as simple API wrappers, **luME** is a low-level systems runtime that coordinates, secures, and executes untrusted agentic code. 

It addresses the critical real-world problem of **AI Safety and Reliability in Autonomous Code Execution**—preventing AI models from running malicious operations or freezing host systems while enabling them to dynamically self-correct their own bugs.

---

## 🏗️ System Architecture & Execution Lifecycle

The runtime processes user requests through a stateful, cyclic directed graph built using **LangGraph**:

```
                     [ User Request ]
                            │
                            ▼
          [ Stage 1: PyTorch Intent Classifier ]
                            │
                            ▼
              [ Stage 2: LangGraph Router ]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      [ RAG Context ]            [ Multi-LLM Debate ]
              │                           │
              ▼                           ▼
      [ Local Search ]           [ Stage 4: AST Checker ]
                                          │
                                          ▼
                               [ Stage 5: Env Manager ]
                               (Dynamic virtualenv setup)
                                          │
                                          ▼
                               [ Stage 6: Sandbox Exec ]
                               (POSIX Limits + Autoencoder)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                 (Error)                                     (Success)
                    ▼                                           ▼
           [ Stage 8: Self-Heal ]                       [ Stage 8: RLSF Update ]
                    │                                           │
                    └───────────────────◄───────────────────────┘
                                        │
                                        ▼
                             [ Stage 9: Telemetry ]
```

### The 9-Stage Pipeline:
1. **Stage 1: Intent Classification (PyTorch NLP Classifier):** A local PyTorch classification network classifies user prompt intent on startup (`DATA_SCIENCE`, `FILE_OPERATION`, `GENERAL_CHAT`).
2. **Stage 2: Stateful Graph Routing (LangGraph):** The orchestrator sets up nodes and conditional routing based on state values.
3. **Stage 3: Decision Voting (Multi-LLM Debate):** Synthesizes code proposals from multiple models (e.g. Llama-3, Gemma via Groq) and uses a referee model to critique and merge them into an optimal script.
4. **Stage 4: Safety & Compiler Checks (AST Sandbox):** Parses the generated Python code into an **Abstract Syntax Tree (AST)**, stripping unsafe libraries (`os`, `subprocess`, `requests`) and raw calls (`open`, `eval`). Safe directory-locked helpers (`read_file`, `write_file`, `list_files`) are injected instead.
5. **Stage 5: Dynamic Env Setup (Virtualenv Manager):** Scans the code imports for dependencies (like `pandas`, `scikit-learn`), automatically builds a nested Python virtual environment (`venv`) inside the sandbox, and installs requirements dynamically.
6. **Stage 6: Shielded Sandbox (POSIX Quotas):** Executes the script with strict system limits (5s CPU limit, 10MB file write size) using the `resource` library.
7. **Stage 7: Live Threat Detection (PyTorch Autoencoder):** Polls the running process metrics (CPU%, Memory%, File writes) in real-time and runs them through a local PyTorch Autoencoder to detect and terminate anomalous behavior.
8. **Stage 8: Self-Healing & Feedback (RLSF Router):** If execution fails, state telemetry loops back to repair the code. On success, feedback goes to our PyTorch RLSF policy router (Policy Gradient RL) to optimize future tool selections.
9. **Stage 9: Event Telemetry:** Commits transaction logs and performance scores to a persistent SQLite database (`lume_telemetry.db`).

---

## 🛠️ Key Technical Highlights
* **POSIX-Level Resource Partitioning:** Active CPU, memory, and write quotas block infinite loops and memory leaks.
* **Static AST Code Analysis:** Acting as an active compiler-level firewall before execution.
* **Local PyTorch Networks:** Runs local neural networks for intent classification, anomaly detection, and reinforcement learning.
* **Consensus Debate Engine:** Multiple models critique each other's outputs to reduce hallucination rates.

---

## 🚀 Quickstart & Usage

### 1. Setup Environment
```bash
cd agenta-py
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Groq API
```bash
export GROQ_API_KEY="your_api_key_here"
export OPENAI_API_KEY="$GROQ_API_KEY"
export OPENAI_API_BASE="https://api.groq.com/openai/v1"
```

### 3. Run Agent Tasks
```bash
python3 -m lume.cli run "Create a file named squares.txt containing the squares of numbers 1 to 5"
```

### 4. View Telemetry Transaction Logs
```bash
python3 -m lume.cli logs
```

---

## 🔮 Future Research Roadmap
1. **Reinforcement Learning from Sandbox Feedback (RLSF):** Training local policy models dynamically via Policy Gradients using exit-code rewards (0 = +1, non-zero = -1) to select tool parameters.
2. **In-Process GGUF LLM Execution:** Replacing external API calls entirely by running quantized models (e.g. Llama-3-8B) inside the python runtime using `llama-cpp-python` with hardware-accelerated local metal/CUDA backends.
3. **Hardware-Isolated Micro-VMs:** Porting execution workspaces from python subprocesses to isolated Firecracker micro-VMs or local WebAssembly (WASM) runtimes for absolute sandbox security.
