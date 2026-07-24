"""Monitoramento de utilização da GPU NVIDIA via pynvml.

Fornece a classe `GPUMonitor` que amostra a utilização da GPU em
background e permite throttling dinâmico para manter a carga próxima
de um alvo configurável (ex: 60%).

Se não houver GPU NVIDIA ou o pynvml não conseguir inicializar, opera
em modo "no-op" — todas as chamadas são inertes.
"""

import time
import logging
import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Optional

from config import GPU_TARGET_PERCENT, GPU_MAX_PERCENT, GPU_CHECK_INTERVAL

logger = logging.getLogger(__name__)


# ─── Data class de resultado ─────────────────────────────────────────────

@dataclass
class EstadoGPU:
    """Estado amostrado da GPU no momento da leitura."""
    disponivel: bool
    utilizacao_percent: float   # 0–100
    memoria_usada_mb: float
    memoria_total_mb: float
    temperatura_c: float


# ─── Classe principal ────────────────────────────────────────────────────

class GPUMonitor:
    """Amostra a utilização da GPU em background e expõe controle de throttling.

    Uso típico:

        monitor = GPUMonitor()
        monitor.iniciar()

        while processando:
            monitor.throttle_se_necessario(max_percent=70)
            # ... submete trabalho para a GPU ...

        monitor.parar()

    Se não houver GPU, todas as operações são no-op seguras.
    """

    def __init__(
        self,
        target_percent: int = GPU_TARGET_PERCENT,
        max_percent: int = GPU_MAX_PERCENT,
        intervalo: float = GPU_CHECK_INTERVAL,
    ) -> None:
        self._target = target_percent
        self._max = max_percent
        self._intervalo = intervalo

        self._disponivel = False
        self._handle: any = None
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._rodando = False

        # Último estado amostrado
        self._utilizacao: float = 0.0
        self._mem_usada: float = 0.0
        self._mem_total: float = 0.0
        self._temp: float = 0.0

        self._inicializar_pynvml()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def disponivel(self) -> bool:
        return self._disponivel

    def iniciar(self) -> None:
        """Inicia a thread de amostragem em background."""
        if not self._disponivel or self._rodando:
            return
        self._rodando = True
        self._thread = Thread(target=self._loop_amostragem, daemon=True)
        self._thread.start()

    def parar(self) -> None:
        """Para a thread de amostragem."""
        self._rodando = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._desinicializar()

    def get_utilizacao(self) -> float:
        """Retorna a última utilização da GPU amostrada (0–100)."""
        return self._utilizacao

    def get_estado(self) -> EstadoGPU:
        """Retorna um snapshot completo do último estado amostrado."""
        with self._lock:
            return EstadoGPU(
                disponivel=self._disponivel,
                utilizacao_percent=self._utilizacao,
                memoria_usada_mb=self._mem_usada,
                memoria_total_mb=self._mem_total,
                temperatura_c=self._temp,
            )

    def throttle_se_necessario(
        self, max_percent: Optional[int] = None
    ) -> None:
        """Se a utilização da GPU exceder *max_percent*, dorme um pouco.

        O sleep é proporcional ao excesso: quanto mais acima do limite,
        mais tempo dorme. Se estiver abaixo do target, não dorme nada.
        """
        if not self._disponivel:
            return
        limite = max_percent if max_percent is not None else self._max
        util = self._utilizacao

        if util > limite:
            excesso = (util - limite) / 100.0
            dormir = min(excesso * 2.0, 1.0)  # max 1s
            time.sleep(dormir)
        elif util < self._target * 0.5:
            # Abaixo de 50% do target → acelera (não dorme)
            pass

    # ------------------------------------------------------------------
    # Inicialização do pynvml
    # ------------------------------------------------------------------

    def _inicializar_pynvml(self) -> None:
        """Tenta conectar ao NVML. Falha silenciosamente se não houver GPU."""
        try:
            import pynvml
            pynvml.nvmlInit()
            qtd = pynvml.nvmlDeviceGetCount()
            if qtd == 0:
                logger.info("GPUMonitor: Nenhum dispositivo NVIDIA encontrado.")
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._disponivel = True
            nome = pynvml.nvmlDeviceGetName(self._handle)
            logger.info(f"GPUMonitor: GPU detectada — {nome}")
        except Exception as exc:
            logger.info(f"GPUMonitor: Não foi possível acessar GPU — {exc}")
            self._disponivel = False

    def _desinicializar(self) -> None:
        """Fecha a conexão NVML."""
        if self._disponivel:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Loop de amostragem
    # ------------------------------------------------------------------

    def _loop_amostragem(self) -> None:
        """Thread loop: amostra a GPU a cada *intervalo* segundos."""
        warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
        import pynvml
        while self._rodando:
            try:
                with self._lock:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._utilizacao = float(util.gpu)

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    self._mem_usada = mem_info.used / (1024 ** 2)
                    self._mem_total = mem_info.total / (1024 ** 2)

                    self._temp = float(
                        pynvml.nvmlDeviceGetTemperature(
                            self._handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
            except Exception:
                pass
            time.sleep(self._intervalo)
