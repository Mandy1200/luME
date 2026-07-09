import typer
import os
import time
from typing import Optional
from lume.config_parser import load_agent_config
from lume.semantic_fs import start_watcher, semantic_search
from lume.orchestrator import execute_agent_task
from lume.telemetry import fetch_telemetry_logs

app = typer.Typer(help="luME: Secure Cognitive OS Sandbox CLI")

@app.command()
def run(
    task: str = typer.Argument(..., help="The task you want the agent to accomplish."),
    config_file: str = typer.Option("config/default_agent.yaml", "--config", "-c", help="Path to the agent YAML config."),
    sandbox_dir: str = typer.Option("sandbox_workspace", "--sandbox-dir", "-s", help="Folder where agent scripts run safely.")
):
    """
    Executes a task using the Agent config. Combines Multi-LLM debate, AST Sandboxing, and Self-Healing.
    """
    if not os.path.exists(config_file):
        typer.echo(f"❌ Config file not found at: {config_file}")
        raise typer.Exit(code=1)
        
    typer.echo("📖 Loading agent configuration...")
    agent = load_agent_config(config_file)
    typer.echo(f"🤖 Loaded Agent '{agent.name}' (using models: {', '.join(agent.models)})")
    
    # Run graph orchestrator
    state = execute_agent_task(
        task=task,
        models=agent.models,
        system_prompt=agent.system_prompt,
        sandbox_dir=sandbox_dir,
        timeout_seconds=agent.sandbox_config.get("timeout_seconds", 5)
    )
    
    success = state["success"]
    output = state["output"]
    final_code = state["code"]
    intent = state["intent"]
    allowed_tools = state["allowed_tools"]
    run_count = state["run_count"]
    
    typer.echo(f"\n🔮 Intent Classified: {intent}")
    typer.echo(f"🛡️ Sandbox Tool Permissions: {allowed_tools}")
    typer.echo(f"🔄 Total Attempts: {run_count}/3")
    
    if success:
        typer.echo("\n🎉 --- SUCCESS ---")
        typer.echo(f"📝 Sandbox Stdout Output:\n{output}")
        typer.echo("\n💻 Final Executed Code:")
        typer.echo(f"```python\n{final_code}\n```")
    else:
        typer.echo("\n💥 --- FAILURE ---")
        typer.echo(f"❌ Error Output:\n{output}")
        typer.echo("\n💻 Final (Broken) Code attempted:")
        typer.echo(f"```python\n{final_code}\n```")


@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic query to search indexed files."),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of matching passages to return.")
):
    """
    Semantically search indexed local files.
    """
    typer.echo(f"🔍 Searching local index for: '{query}'...")
    results = semantic_search(query, top_k=top_k)
    
    if not results:
        typer.echo("ℹ️ No documents indexed yet or no matches found.")
        return
        
    for idx, hit in enumerate(results):
        typer.echo(f"\n[{idx+1}] 📄 {hit['filepath']} (Similarity: {hit['similarity']:.4f})")
        typer.echo("-" * 40)
        typer.echo(hit['content'].strip())
        typer.echo("-" * 40)


@app.command()
def watch(
    directory: str = typer.Option(".", "--dir", "-d", help="Directory to sync and watch for changes.")
):
    """
    Starts watching a folder, embedding and indexing modified files in real-time.
    """
    typer.echo(f"👀 Indexing and watching folder: {os.path.abspath(directory)}")
    observer = start_watcher(directory)
    
    typer.echo("✨ Background watcher running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\n🛑 Stopping background watcher...")
        observer.stop()
    observer.join()
    typer.echo("👋 Watcher shut down.")


@app.command()
def logs(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of telemetry logs to display.")
):
    """
    Displays the state transaction logs from the SQLite telemetry database.
    """
    typer.echo("📊 Fetching luME state transaction logs...")
    log_rows = fetch_telemetry_logs()
    
    if not log_rows:
        typer.echo("ℹ️ No logs recorded yet.")
        return
        
    for row in log_rows[:limit]:
        log_id, ts, session, node, status, msg, metrics = row
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
        typer.echo(f"\n[{time_str}] Session: {session[:8]}... Node: {node}")
        typer.echo(f"  Status: {status} | Msg: {msg}")
        if metrics and metrics != "{}":
            typer.echo(f"  Metrics: {metrics}")


if __name__ == "__main__":
    app()
