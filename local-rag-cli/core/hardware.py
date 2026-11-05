"""
Local RAG CLI - Hardware Profiler & Inference Safety Engine
Analyzes system hardware (CPU, RAM, GPU) with zero external dependencies
using Windows native APIs and evaluates model memory safety and performance fit.
"""

import os
import sys
import platform
import shutil
import subprocess
from typing import Dict, Any, List, Optional

# In-memory cache for hardware detection to avoid redundant queries
_HARDWARE_CACHE: Optional[Dict[str, Any]] = None


def get_cpu_simd_feature() -> str:
    """
    Detects the highest vector SIMD instruction set supported by the host CPU.
    Uses Windows native IsProcessorFeaturePresent API via kernel32:
      - 49 = PF_AVX512F_INSTRUCTIONS_AVAILABLE
      - 40 = PF_AVX2_INSTRUCTIONS_AVAILABLE
      - 25 = PF_AVX_INSTRUCTIONS_AVAILABLE
    """
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            if k32.IsProcessorFeaturePresent(49):
                return "AVX-512"
            if k32.IsProcessorFeaturePresent(40):
                return "AVX2"
            if k32.IsProcessorFeaturePresent(25):
                return "AVX"
        except Exception:
            pass
    return "Standard"


def get_cpu_info() -> Dict[str, Any]:
    """Retrieves CPU model name, logical core count, and highest SIMD instruction set."""
    cores = os.cpu_count() or 4
    cpu_name = platform.processor() or "Generic x86_64 Processor"

    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if val and val.strip():
                    cpu_name = val.strip()
        except Exception:
            pass

    return {
        "cores": cores,
        "model": cpu_name,
        "simd": get_cpu_simd_feature()
    }


def get_memory_info() -> Dict[str, float]:
    """
    Retrieves total and available physical RAM in Gigabytes.
    Uses Windows GlobalMemoryStatusEx via ctypes for sub-millisecond precision.
    """
    total_gb = 8.0
    avail_gb = 4.0

    if sys.platform == "win32":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_gb = round(stat.ullTotalPhys / (1024 ** 3), 1)
                avail_gb = round(stat.ullAvailPhys / (1024 ** 3), 1)
        except Exception:
            pass

    return {
        "total_gb": total_gb,
        "available_gb": avail_gb
    }


def get_gpu_info() -> Dict[str, Any]:
    """
    Discovers installed GPUs.
    Probes for dedicated NVIDIA CUDA GPU (with VRAM) or integrated GPUs (Intel/AMD).
    """
    gpus: List[str] = []
    has_dedicated = False
    dedicated_name = None
    vram_gb: Optional[float] = None

    # 1. Probe for NVIDIA GPU via nvidia-smi if available in PATH
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 1:
                        name = parts[0].strip()
                        gpus.append(name)
                        has_dedicated = True
                        dedicated_name = name
                    if len(parts) >= 2:
                        try:
                            mb = float(parts[1].strip())
                            vram_gb = round(mb / 1024, 1)
                        except ValueError:
                            pass
        except Exception:
            pass

    # 2. Windows registry probe for display adapters (NVIDIA, AMD Radeon RX, Intel Arc, etc.)
    if sys.platform == "win32":
        try:
            import winreg
            base_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_key) as root_key:
                for idx in range(16):
                    try:
                        sub_name = winreg.EnumKey(root_key, idx)
                        with winreg.OpenKey(root_key, sub_name) as sub_key:
                            try:
                                desc, _ = winreg.QueryValueEx(sub_key, "DriverDesc")
                                if desc and desc.strip() and desc.strip() not in gpus:
                                    gpus.append(desc.strip())
                                    lower_d = desc.lower()

                                    # Check for dedicated GPUs across all major vendors
                                    is_nvidia_ded = any(k in lower_d for k in ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"])
                                    is_amd_ded = any(k in lower_d for k in ["radeon rx", "radeon pro", "firepro", "rx 6", "rx 7", "rx 5"])
                                    is_intel_ded = any(k in lower_d for k in ["arc(tm)", "arc a", "intel arc", "arc pro"])

                                    if is_nvidia_ded or is_amd_ded or is_intel_ded:
                                        has_dedicated = True
                                        dedicated_name = desc.strip()

                                        # Query total physical dedicated VRAM if not already retrieved
                                        if vram_gb is None:
                                            try:
                                                qw_vram, _ = winreg.QueryValueEx(sub_key, "HardwareInformation.qwMemorySize")
                                                vram_gb = round(qw_vram / (1024 ** 3), 1)
                                            except FileNotFoundError:
                                                try:
                                                    raw_mem, _ = winreg.QueryValueEx(sub_key, "HardwareInformation.MemorySize")
                                                    if isinstance(raw_mem, int):
                                                        vram_gb = round(raw_mem / (1024 ** 3), 1)
                                                    elif isinstance(raw_mem, bytes):
                                                        vram_int = int.from_bytes(raw_mem, byteorder="little")
                                                        vram_gb = round(vram_int / (1024 ** 3), 1)
                                                except Exception:
                                                    pass
                            except FileNotFoundError:
                                pass
                    except OSError:
                        break
        except Exception:
            pass

    primary_gpu = gpus[0] if gpus else "Integrated Display Adapter"

    simd = get_cpu_simd_feature()
    if has_dedicated and dedicated_name:
        vram_str = f" ({vram_gb} GB VRAM)" if vram_gb else ""
        if "nvidia" in (dedicated_name or "").lower() or "geforce" in (dedicated_name or "").lower():
            device_mode = f"CUDA GPU Offload [{dedicated_name}{vram_str}]"
        else:
            device_mode = f"Dedicated GPU [{dedicated_name}{vram_str}]"
    else:
        simd_str = f" ({simd} Vector Mode)" if simd != "Standard" else ""
        device_mode = f"CPU Inference{simd_str}"

    return {
        "gpus": gpus,
        "primary_gpu": primary_gpu,
        "has_dedicated_gpu": has_dedicated,
        "dedicated_name": dedicated_name,
        "vram_gb": vram_gb,
        "device_mode": device_mode
    }


def get_system_hardware_info(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Returns complete consolidated hardware profile for the host system.
    Results are cached in memory for zero subsequent performance overhead.
    """
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None and not force_refresh:
        # Update dynamic available RAM while keeping static hardware cached
        mem = get_memory_info()
        _HARDWARE_CACHE["available_ram_gb"] = mem["available_gb"]
        return _HARDWARE_CACHE

    cpu = get_cpu_info()
    mem = get_memory_info()
    gpu = get_gpu_info()

    # Determine Hardware Tier & Recommendation
    total_ram = mem["total_gb"]
    if gpu["has_dedicated_gpu"] and (gpu["vram_gb"] or 0) >= 4.0:
        tier = "Tier 3: Dedicated GPU"
        recommended_range = "3B - 8B Models (CUDA Accelerated)"
    elif total_ram >= 12.0:
        tier = "Tier 2: Balanced CPU"
        recommended_range = "1.5B - 3.8B Models (CPU Optimized)"
    else:
        tier = "Tier 1: Ultra-Light CPU"
        recommended_range = "1B - 1.5B Models (Safe Low-Memory)"

    profile = {
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "total_ram_gb": mem["total_gb"],
        "available_ram_gb": mem["available_gb"],
        "gpus": gpu["gpus"],
        "primary_gpu": gpu["primary_gpu"],
        "has_dedicated_gpu": gpu["has_dedicated_gpu"],
        "dedicated_name": gpu["dedicated_name"],
        "vram_gb": gpu["vram_gb"],
        "device_mode": gpu["device_mode"],
        "hardware_tier": tier,
        "recommended_range": recommended_range,
    }

    _HARDWARE_CACHE = profile
    return profile


def estimate_model_ram_requirement(model_name: str) -> float:
    """
    Estimates RAM / VRAM footprint (in GB) required for 4-bit / Q4 quantized weights + KV cache.
    """
    lower = model_name.lower()
    if "0.5b" in lower:
        return 0.8
    if "1.5b" in lower or "1b" in lower or "tiny" in lower:
        return 1.4
    if "3.8b" in lower or "phi3" in lower or "phi-3" in lower or "3b" in lower:
        return 2.6
    if "7b" in lower or "8b" in lower or "llama3" in lower or "mistral" in lower:
        return 5.5
    if "13b" in lower or "14b" in lower:
        return 9.8
    if "32b" in lower or "33b" in lower:
        return 20.0
    if "70b" in lower:
        return 42.0

    # Default fallback for unknown model names
    return 3.0


def evaluate_model_safety(model_name: str, hw_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluates whether a model is safe to run on current hardware without
    causing system thrashing, swap paging, or Out-Of-Memory freezing.
    """
    if hw_info is None:
        hw_info = get_system_hardware_info()

    est_ram = estimate_model_ram_requirement(model_name)
    has_dedicated = hw_info["has_dedicated_gpu"]
    vram_gb = hw_info["vram_gb"] or 0.0
    total_ram = hw_info["total_ram_gb"]
    avail_ram = hw_info["available_ram_gb"]

    # Evaluation on Dedicated GPU
    if has_dedicated and vram_gb >= est_ram:
        return {
            "model_name": model_name,
            "est_ram_gb": est_ram,
            "safety_level": "EXCELLENT",
            "badge": "[bold green]GPU-ACCELERATED - EXCELLENT[/bold green]",
            "menu_badge": "[✔ GPU Accelerated • High Speed]",
            "warning": None,
            "is_safe": True
        }

    # Evaluation on CPU
    # 1. Ultra safe: fits easily in RAM
    if est_ram <= 1.8:
        return {
            "model_name": model_name,
            "est_ram_gb": est_ram,
            "safety_level": "EXCELLENT",
            "badge": "[bold green]CPU-SAFE - EXCELLENT FIT[/bold green]",
            "menu_badge": "[✔ Recommended for CPU • Fast & Light]",
            "warning": None,
            "is_safe": True
        }

    # 2. Moderate CPU models (e.g. Phi-3 3.8B on 12-16GB RAM)
    if est_ram <= 3.2 and total_ram >= 12.0:
        warning_msg = None
        if avail_ram < 2.0:
            warning_msg = f"Available RAM is {avail_ram} GB. Phi-3 requires ~{est_ram} GB. Windows may page memory if heavy apps are open."
        return {
            "model_name": model_name,
            "est_ram_gb": est_ram,
            "safety_level": "GOOD" if not warning_msg else "CAUTION",
            "badge": "[bold cyan]CPU-SAFE - HIGH REASONING[/bold cyan]" if not warning_msg else "[bold yellow]CPU-SAFE - TIGHT RAM[/bold yellow]",
            "menu_badge": "[✔ Supported • High Reasoning • Moderate CPU Speed]",
            "warning": warning_msg,
            "is_safe": True
        }

    # 3. Heavy models on CPU (e.g. 7B - 8B on CPU)
    if est_ram <= 6.5:
        warning_msg = (
            f"Model '{model_name}' requires ~{est_ram} GB RAM. "
            "On pure CPU inference, token generation speed may slow to 2-4 tok/s "
            "and may cause memory pressure."
        )
        return {
            "model_name": model_name,
            "est_ram_gb": est_ram,
            "safety_level": "CAUTION",
            "badge": "[bold yellow]CAUTION - HEAVY CPU FOOTPRINT[/bold yellow]",
            "menu_badge": "[⚠️ Heavy on CPU • Slower Generation (~3 tok/s)]",
            "warning": warning_msg,
            "is_safe": True  # Will run, but user must be warned
        }

    # 4. Critical models (14B+ on standard consumer CPU)
    warning_msg = (
        f"Model '{model_name}' requires ~{est_ram} GB RAM (Exceeds safe limits). "
        "Running this model on CPU will likely freeze or severely lag your system."
    )
    return {
        "model_name": model_name,
        "est_ram_gb": est_ram,
        "safety_level": "CRITICAL",
        "badge": "[bold red]CRITICAL • EXCEEDS HARDWARE SAFE CAPACITY[/bold red]",
        "menu_badge": "[⛔ Not Recommended • Exceeds System RAM]",
        "warning": warning_msg,
        "is_safe": False
    }
