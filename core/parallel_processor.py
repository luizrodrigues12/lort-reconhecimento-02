"""Processamento paralelo de fotos com alvo de ~100% de uso da CPU.

Usa ``ThreadPoolExecutor`` com sliding window para paralelizar a
inferência mantendo memória constante. O ONNX Runtime libera o GIL
durante a execução, permitindo paralelismo real.

O buffer de exibição ordenada garante que os logs apareçam
sequencialmente (1, 2, 3...) mesmo quando as fotos terminam em
ordem diferente.

Shutdown (CTRL+C / SIGTERM)
----------------------------
``ShutdownManager`` captura o sinal e encerra o processamento
graciosamente, sem tracebacks.
"""

import signal
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Dict, List, Optional

import numpy as np

from config import (
    CPU_TARGET_PERCENT,
    CACHE_INTERVAL,
    MIN_WORKERS,
    MAX_WORKERS,
    PARALLEL_MIN_PHOTOS,
)
from core.pipeline_worker import processar_unidade
from core.centrality import AnalisadorFoco
from services.cache_service import CacheService
from services.face_engine import FaceEngine


# ══════════════════════════════════════════════════════════════════════════
#  ShutdownManager  —  CTRL+C / SIGTERM com confirmação
# ══════════════════════════════════════════════════════════════════════════


class ShutdownManager:
    """Captura SIGINT/SIGTERM e encerra imediatamente.

    Uso:
        shutdown = ShutdownManager()
        shutdown.instalar()

        for ...:
            if shutdown.confirmar_ou_continuar():
                break   # ➜ CTRL+C detectado, encerramentro limpo
    """

    _instance: Optional["ShutdownManager"] = None

    def __init__(self) -> None:
        self._interrupted = False
        self._notified = False
        self._lock = threading.Lock()

    @classmethod
    def instancia(cls) -> "ShutdownManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── API pública ─────────────────────────────────────────────────

    def instalar(self) -> None:
        """Registra os handlers de SIGINT e SIGTERM."""
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    @property
    def is_requested(self) -> bool:
        return self._interrupted

    def confirmar_ou_continuar(self) -> bool:
        """Retorna True se CTRL+C foi pressionado. Sem perguntas."""
        if self._interrupted:
            if not self._notified:
                self._notified = True
                print("\n\n👋 Processamento interrompido pelo usuário — encerrando...")
            return True
        return False

    # ── Signal handler ──────────────────────────────────────────────

    def _on_signal(self, signum: int, frame) -> None:
        """Handler do sinal — apenas seta a flag."""
        with self._lock:
            self._interrupted = True


# ══════════════════════════════════════════════════════════════════════════
#  ParallelProcessor  —  classe principal
# ══════════════════════════════════════════════════════════════════════════


class ParallelProcessor:
    """Processa fotos em paralelo visando ~100% de uso da CPU.

    Usa ``ThreadPoolExecutor`` com N threads + sliding window para
    manter memória constante. O modelo InsightFace é compartilhado
    entre todas as threads; ONNX Runtime libera o GIL durante a
    inferência, permitindo paralelismo real.

    Parâmetros
    ----------
    face_engine : FaceEngine
        Instância do motor facial.
    cache_service : CacheService
        Serviço de cache.
    pasta_destino : str
        Raiz da pasta de destino.
    """

    def __init__(
        self,
        face_engine: "FaceEngine",
        cache_service: "CacheService",
        pasta_destino: str,
        mover: bool = False,
        separar_grupos: bool = True,
        criar_pasta_nao_reconhecidos: bool = True,
    ) -> None:
        self._face_engine = face_engine
        self._cache_service = cache_service
        self._pasta_destino = pasta_destino
        self._mover = mover
        self._separar_grupos = separar_grupos
        self._criar_pasta_nao_reconhecidos = criar_pasta_nao_reconhecidos

        # Acumuladores (agregados dos workers / do loop)
        self._contadores: Dict[str, int] = {}
        self._total_grupos = 0
        self._total_nao_reconhecidos = 0
        self._total_cache = 0
        self._total_rostos = 0
        self._cache_updates: Dict[str, str] = {}

        # Buffer de exibição ordenada (logs sempre na sequência 1,2,3...)
        self._buffer: Dict[int, dict] = {}
        self._proximo = 1
        self._display_lock = threading.Lock()

        # Analisador de foco (única instância compartilhada)
        self._analisador = AnalisadorFoco()

        # Shutdown manager (singleton)
        self._shutdown = ShutdownManager.instancia()

    # ── API pública ─────────────────────────────────────────────────

    def processar(
        self,
        fotos: List[Path],
        matriz_refs: np.ndarray,
        nomes_refs: List[str],
    ) -> Dict:
        """Pipeline principal — delega para CPU, GPU ou serial conforme
        a quantidade de fotos e o modo selecionado."""
        if not fotos:
            return {}

        total = len(fotos)
        print(f"- {total} FOTO(S) encontrada(s).\n")
        print("─" * 60)

        self._contadores = {nome: 0 for nome in nomes_refs}

        # Decide estratégia: serial para poucas, paralelo para muitas
        if total < PARALLEL_MIN_PHOTOS:
            resultado = self._processar_serial(fotos, matriz_refs, nomes_refs)
        else:
            resultado = self._processar_cpu(fotos, matriz_refs, nomes_refs)

        # Salva cache final
        self._despejar_cache()
        self._cache_service.salvar_final()

        return resultado

    # ── CPU mode  (ThreadPoolExecutor) ─────────────────────────────

    def _processar_cpu(
        self,
        fotos: List[Path],
        matriz_refs: np.ndarray,
        nomes_refs: List[str],
    ) -> Dict:
        n_workers = self._calcular_workers()
        print(
            f"  ⚙️  {n_workers} thread(s) paralelas "
            f"(alvo: {CPU_TARGET_PERCENT}% CPU)\n"
        )

        app = self._face_engine.app
        cache_snapshot = self._cache_service.snapshot()
        total = len(fotos)

        # Janela deslizante: no máximo n_workers * 2 tarefas na fila.
        # Elimina degradação O(n²) do wait() e evita acúmulo de memória.
        window_size = max(1, n_workers * 2)
        foto_iter = iter(enumerate(fotos, 1))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futuros: Dict = {}

            def _submeter():
                try:
                    i, foto = next(foto_iter)
                    fut = executor.submit(
                        processar_unidade,
                        foto, matriz_refs, nomes_refs,
                        self._pasta_destino, app, cache_snapshot,
                        i, total, self._analisador,
                        self._mover, self._separar_grupos,
                        self._criar_pasta_nao_reconhecidos,
                    )
                    futuros[fut] = foto
                except StopIteration:
                    pass

            # Lote inicial
            for _ in range(min(window_size, total)):
                _submeter()

            while futuros:
                if self._shutdown.is_requested:
                    if self._shutdown.confirmar_ou_continuar():
                        break

                done, _ = wait(
                    futuros.keys(), timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )

                for futuro in done:
                    try:
                        resultado = futuro.result()
                    except OSError as exc:
                        foto = futuros.get(futuro)
                        self._tratar_erro_worker(foto, exc)
                        if getattr(exc, "errno", 0) == 28:
                            print("  ⚠️  Disco cheio — interrompendo processamento.")
                            break
                    except Exception as exc:
                        foto = futuros.get(futuro)
                        self._tratar_erro_worker(foto, exc)
                    else:
                        self._acumular(resultado)
                        self._exibir_ordenado(resultado)
                    finally:
                        futuros.pop(futuro, None)
                        _submeter()
                else:
                    if self._shutdown.is_requested:
                        if self._shutdown.confirmar_ou_continuar():
                            break
                    continue
                break

        return self._montar_resultado()

    # ── Serial mode (poucas fotos, overhead do pool não compensa) ──

    def _processar_serial(
        self,
        fotos: List[Path],
        matriz_refs: np.ndarray,
        nomes_refs: List[str],
    ) -> Dict:
        print("  ℹ️  Poucas fotos — processamento serial\n")
        app = self._face_engine.app
        cache_snapshot = self._cache_service.snapshot()
        total = len(fotos)

        for i, foto in enumerate(fotos, 1):
            if self._shutdown.is_requested:
                if self._shutdown.confirmar_ou_continuar():
                    break
            try:
                resultado = processar_unidade(
                    foto,
                    matriz_refs,
                    nomes_refs,
                    self._pasta_destino,
                    app,
                    cache_snapshot,
                    i,
                    total,
                    self._analisador,
                    self._mover,
                    self._separar_grupos,
                    self._criar_pasta_nao_reconhecidos,
                )
            except OSError as exc:
                self._tratar_erro_worker(foto, exc)
                if getattr(exc, "errno", 0) == 28:
                    print("  ⚠️  Disco cheio — interrompendo processamento.")
                    break
                continue
            self._acumular(resultado)
            self._exibir_ordenado(resultado)

        return self._montar_resultado()

    # ── Helpers internos ───────────────────────────────────────────

    def _calcular_workers(self) -> int:
        """Calcula o número de threads para atingir ~60% de CPU alvo.

        Fórmula: ``workers = round(núcleos * alvo / 100)`` com piso em
        ``MIN_WORKERS`` e teto em ``MAX_WORKERS`` e no total de núcleos.
        """
        import psutil

        nucleos = psutil.cpu_count(logical=True)
        workers = max(MIN_WORKERS, round(nucleos * CPU_TARGET_PERCENT / 100))
        return min(workers, MAX_WORKERS, nucleos)

    def _acumular(self, resultado: Dict) -> None:
        """Agrega o resultado de uma foto nos contadores e no cache.

        Chamado sempre da **thread principal** (o ``as_completed`` e os
        loops serial/GPU rodam nela), portanto não há concorrência aqui.
        """
        if resultado.get("cache"):
            self._total_cache += 1
        elif resultado.get("erro"):
            self._total_nao_reconhecidos += 1
        elif resultado.get("categoria") == "Grupos":
            self._total_grupos += 1
        elif resultado.get("categoria") == "Nao Reconhecidos":
            self._total_nao_reconhecidos += 1
        elif resultado.get("categoria"):
            nome = resultado["categoria"]
            self._contadores[nome] = self._contadores.get(nome, 0) + 1

        self._total_rostos += resultado.get("qtd_rostos", 0)

        # Cache tracking: acumula updates para despejar periódicamente
        caminho = resultado.get("caminho", "")
        hash_val = resultado.get("hash")
        if caminho and hash_val and not resultado.get("cache"):
            self._cache_updates[caminho] = hash_val
            if len(self._cache_updates) >= CACHE_INTERVAL:
                self._despejar_cache()

    def _exibir_ordenado(self, resultado: Dict) -> None:
        """Exibe o log do resultado na ordem correta (1,2,3...).

        Buffereiza resultados que chegam fora de ordem e os imprime
        assim que o proximo indice esperado estiver disponivel.
        Thread-safe via _display_lock.
        """
        with self._display_lock:
            self._buffer[resultado["_indice"]] = resultado
            while self._proximo in self._buffer:
                r = self._buffer.pop(self._proximo)
                if r.get("_log_msg"):
                    print(r["_log_msg"])
                self._proximo += 1

    def _despejar_cache(self) -> None:
        """Transfere as atualizações acumuladas para o CacheService."""
        for caminho, h in self._cache_updates.items():
            self._cache_service.marcar_processada(caminho, h)
        self._cache_updates.clear()
        self._cache_service.salvar_incrementalmente()

    def _tratar_erro_worker(self, foto: Path, exc: Exception) -> None:
        """Log de erro inesperado de um worker."""
        print(f"\n  ⚠️  Erro ao processar '{foto.name}': {exc}")
        self._total_nao_reconhecidos += 1

    def _montar_resultado(self) -> Dict:
        """Monta o dicionário de resultado no formato esperado por main.py
        (compatível com o antigo ``Acumulador.para_dict``)."""
        return {
            "contadores": self._contadores,
            "total_grupos": self._total_grupos,
            "total_nao_reconhecidos": self._total_nao_reconhecidos,
            "total_cache": self._total_cache,
            "total_rostos": self._total_rostos,
        }
