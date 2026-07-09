import os
from typing import List, Dict, Any
from openai import OpenAI

def get_client() -> OpenAI:
    # Use OpenAI API key if available; otherwise default to local Ollama API
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
    return OpenAI(api_key=api_key, base_url=base_url)

def query_model(model_name: str, system_prompt: str, user_prompt: str) -> str:
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: Failed to query model {model_name}. Details: {e}"

def extract_python_code(content: str) -> str:
    """
    Strips markdown code blocks if the model wrapped the code.
    """
    if "```python" in content:
        parts = content.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    return content.strip()

def run_debate(models: List[str], system_prompt: str, task: str) -> str:
    """
    Queries multiple models to get proposals, then does a consensus/debate step
    where they critique and produce a single merged solution.
    """
    if not models:
        raise ValueError("At least one model must be specified.")
    
    # Step 1: Get initial proposals from all models
    proposals = {}
    for model in models:
        print(f"🤖 Querying model '{model}' for code proposal...")
        response = query_model(model, system_prompt, task)
        code = extract_python_code(response)
        if not code.startswith("ERROR:"):
            proposals[model] = code
            print(f"✅ Received proposal from '{model}' ({len(code)} chars)")
        else:
            print(f"❌ '{model}' failed: {code}")
            
    if not proposals:
        raise RuntimeError("All models failed to generate a proposal.")
    
    # If only one model was queried or succeeded, return its proposal directly.
    if len(proposals) == 1:
        return list(proposals.values())[0]
        
    # Step 2: Debate step
    print("⚖️ Running Consensus Debate step between models...")
    debate_prompt = "Compare the following proposed python solutions for this task: '" + task + "'\n\n"
    for idx, (model, code) in enumerate(proposals.items()):
        debate_prompt += f"--- PROPOSAL {idx+1} (from model: {model}) ---\n{code}\n\n"
        
    debate_prompt += (
        "Identify any bugs, omissions, or syntax errors in these proposals. "
        "Combine the best parts of both into a single, highly optimized, and clean Python script. "
        "Provide ONLY the final Python code. Do not include explanation text."
    )
    
    # Let the first model act as the final Referee/Broker
    referee = models[0]
    print(f"🧠 Refereeing consensus with model '{referee}'...")
    final_response = query_model(referee, "You are a senior software architect refereeing a consensus debate.", debate_prompt)
    
    return extract_python_code(final_response)
