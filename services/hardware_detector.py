"""Detecção de GPU/CUDA e escolha do modo de execução (GPU vs CPU)."""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


# ─── Diagnóstico de hardware ──────────────────────────────────────────────

@dataclass
class DiagnosticoHardware:
    """Resultado completo da detecção: o que foi encontrado e o que recomendar."""
    gpu_encontrada: bool
    nome_gpu: str
    cuda_disponivel: bool
    modo_recomendado: str      # "gpu" | "cpu"
    resumo: str


def _executar_nvidia_smi() -> tuple[bool, str]:
    """
    Consulta nvidia-smi para detectar GPU NVIDIA e obter o nome.
    Retorna (encontrada, nome_da_gpu).
    """
    try:
        resultado = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if resultado.returncode != 0:
            return False, ""
        nome = resultado.stdout.strip()
        return bool(nome), nome.split("\n")[0] if nome else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def _verificar_cuda_onnx() -> bool:
    """Verifica se o onnxruntime (usado pelo InsightFace) tem suporte a CUDA."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def detectar_hardware() -> DiagnosticoHardware:
    """Detecta GPU NVIDIA + CUDA e retorna o diagnóstico completo."""
    gpu_ok, nome_gpu = _executar_nvidia_smi()
    cuda_ok = _verificar_cuda_onnx() if gpu_ok else False

    if gpu_ok and cuda_ok:
        return DiagnosticoHardware(
            gpu_encontrada=True,
            nome_gpu=nome_gpu,
            cuda_disponivel=True,
            modo_recomendado="gpu",
            resumo=f"GPU detectada ({nome_gpu} + CUDA disponível) — recomendado usar GPU",
        )
    if gpu_ok and not cuda_ok:
        return DiagnosticoHardware(
            gpu_encontrada=True,
            nome_gpu=nome_gpu,
            cuda_disponivel=False,
            modo_recomendado="cpu",
            resumo=f"GPU detectada ({nome_gpu}) mas CUDA não disponível — usando CPU",
        )
    return DiagnosticoHardware(
        gpu_encontrada=False,
        nome_gpu="",
        cuda_disponivel=False,
        modo_recomendado="cpu",
        resumo="CUDA não disponível — usando CPU",
    )


# ─── Escolha do modo de execução ─────────────────────────────────────────

def _modo_de_env() -> Optional[str]:
    """Verifica variável de ambiente FACE_DEVICE."""
    env = os.getenv("FACE_DEVICE", "").strip().lower()
    if env in ("gpu", "cpu"):
        return env
    return None


def escolher_modo_execucao() -> str:
    """Interface com o usuário: detecta hardware, pergunta e retorna 'gpu' ou 'cpu'."""
    # 1. Prioridade máxima: variável de ambiente FACE_DEVICE
    modo_forcado = _modo_de_env()
    if modo_forcado:
        return modo_forcado

    # 2. Detecta hardware
    diag = detectar_hardware()
    print(f"\n🔍 {diag.resumo}")

    # 3. Sem GPU ou sem CUDA → CPU automático
    if diag.modo_recomendado == "cpu":
        print("→ Modo CPU selecionado automaticamente.\n")
        return "cpu"

    # 4. GPU disponível → pergunta ao usuário
    while True:
        escolha = input(
            "\nDeseja usar GPU (CUDA) para acelerar o reconhecimento facial? [S/N]: "
        ).strip().upper()
        if escolha in ("S", "SIM"):
            print("→ Modo GPU selecionado.\n")
            return "gpu"
        if escolha in ("N", "NAO", "NÃO"):
            print("→ Modo CPU selecionado.\n")
            return "cpu"
        print("  Resposta inválida. Digite S (sim) ou N (não).")
