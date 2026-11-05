import json
import urllib.request
import urllib.error
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL, get_active_model


def check_ollama_service(base_url: str = OLLAMA_BASE_URL) -> Dict[str, Any]:
    """
    Checks if Ollama daemon is running and returns available models.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LocalRAG-HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return {
                    "online": True,
                    "models": models,
                    "error": None
                }
            return {
                "online": False,
                "models": [],
                "error": f"Ollama returned HTTP status {resp.status}"
            }
    except urllib.error.URLError as e:
        return {
            "online": False,
            "models": [],
            "error": f"Cannot connect to Ollama at {base_url} ({e.reason})"
        }
    except Exception as e:
        return {
            "online": False,
            "models": [],
            "error": str(e)
        }


def is_model_installed(target_model: str, installed_models: List[str]) -> bool:
    """
    Checks whether target_model exists in installed_models.
    Handles tags like 'phi3' matching 'phi3:latest'.
    """
    target_clean = target_model.strip().lower()
    target_base = target_clean.split(":")[0]

    for installed in installed_models:
        inst_clean = installed.strip().lower()
        inst_base = inst_clean.split(":")[0]
        if target_clean == inst_clean or target_base == inst_base or target_clean == inst_base:
            return True
    return False


def verify_system_readiness(base_url: str = OLLAMA_BASE_URL) -> Dict[str, Any]:
    """
    Performs full health verification for Ollama and required models.
    """
    ollama_status = check_ollama_service(base_url)
    if not ollama_status["online"]:
        return {
            "ready": False,
            "ollama_online": False,
            "embedding_ready": False,
            "llm_ready": False,
            "available_models": [],
            "issues": [
                f"Ollama server is not responding at {base_url}.",
                "Fix: Start Ollama by running 'ollama serve' in a terminal or launching the Ollama desktop app."
            ]
        }

    installed = ollama_status["models"]
    active_llm = get_active_model()
    has_embed = is_model_installed(EMBEDDING_MODEL, installed)
    has_llm = is_model_installed(active_llm, installed)

    issues = []
    if not has_embed:
        issues.append(f"Missing embedding model '{EMBEDDING_MODEL}'. Fix: run 'ollama pull {EMBEDDING_MODEL}'")
    if not has_llm:
        issues.append(f"Missing active LLM model '{active_llm}'. Fix: run 'ollama pull {active_llm}'")

    ready = has_embed and has_llm
    return {
        "ready": ready,
        "ollama_online": True,
        "embedding_ready": has_embed,
        "llm_ready": has_llm,
        "active_model": active_llm,
        "available_models": installed,
        "issues": issues
    }


from core.hardware import get_system_hardware_info, evaluate_model_safety


def display_health_banner(console: Console) -> bool:
    """
    Displays a formatted status panel on startup covering Ollama readiness,
    host hardware specs (CPU/RAM/GPU), and active LLM safety evaluation.
    Returns True if system is completely ready, False otherwise.
    """
    status = verify_system_readiness()
    active_llm = get_active_model()
    hw = get_system_hardware_info()
    safety = evaluate_model_safety(active_llm, hw)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Component", style="bold")
    table.add_column("Status")

    # Ollama status row
    if status["ollama_online"]:
        table.add_row("Ollama Daemon", "[bold green]ONLINE[/bold green] (localhost:11434)")
    else:
        table.add_row("Ollama Daemon", "[bold red]OFFLINE[/bold red]")

    # Embedding Model row
    if status["embedding_ready"]:
        table.add_row(f"Embeddings ({EMBEDDING_MODEL})", "[bold green]READY[/bold green]")
    else:
        table.add_row(f"Embeddings ({EMBEDDING_MODEL})", "[bold red]NOT FOUND[/bold red]")

    # LLM Model row
    if status["llm_ready"]:
        table.add_row(f"Active LLM ({active_llm})", "[bold green]READY[/bold green]")
    else:
        table.add_row(f"Active LLM ({active_llm})", "[bold red]NOT FOUND[/bold red]")

    # Hardware & Inference Profile rows
    mem_avail_str = f"{hw['total_ram_gb']} GB RAM [dim](Available: {hw['available_ram_gb']} GB)[/dim]"
    table.add_row("CPU & Memory", f"{hw['cpu_cores']} Cores | {mem_avail_str}")
    table.add_row("Inference Device", f"[cyan]{hw['device_mode']}[/cyan]")
    table.add_row("Hardware Profile", f"{hw['hardware_tier']} [dim]({hw['recommended_range']})[/dim]")
    table.add_row("Active Model Fit", safety["badge"])

    border_style = "green" if status["ready"] and safety["is_safe"] else "yellow"
    title = "[bold green]System Health & Hardware Engine[/bold green]" if status["ready"] else "[bold yellow]System Health Alert[/bold yellow]"

    console.print(Panel(table, title=title, border_style=border_style, expand=False))

    if not status["ready"]:
        console.print("\n[bold yellow]Attention required before running queries:[/bold yellow]")
        for issue in status["issues"]:
            console.print(f"  • {issue}")
        console.print()

    # Proactive hardware warnings for tight memory or heavy models
    if safety.get("warning"):
        console.print(f"\n[bold yellow]Hardware Advisory for '{active_llm}':[/bold yellow]")
        console.print(f"  • [yellow]{safety['warning']}[/yellow]")
        console.print(f"  • [dim]Tip: You can switch to a faster CPU-safe model via the main menu.[/dim]\n")

    return status["ready"]

