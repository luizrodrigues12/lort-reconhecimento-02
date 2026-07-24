"""Serviço de leitura, decodificação e redimensionamento de imagens.
Com suporte a arquivos RAW via rawpy (NEF, CR2, CR3, ARW, etc.).
"""

import io
import threading
from pathlib import Path

import cv2
import numpy as np
import xxhash

from config import MAX_DIMENSION, RAW_EXTENSIONS, RAW_HALF_SIZE

# Serializa decodificação RAW para evitar contenção de GIL entre threads.
# rawpy.postprocess() não libera o GIL adequadamente; 6 threads competindo
# causam ~5ms de context switch por vez, reduzindo throughput a ~1 fps.
_raw_decode_lock = threading.Lock()


def _is_raw(caminho: str) -> bool:
    """Verifica se o arquivo tem extensão RAW conhecida."""
    return Path(caminho).suffix.lower() in RAW_EXTENSIONS


def ler_bytes_e_hash(caminho: str) -> tuple[bytes, str]:
    """Lê o arquivo uma vez: retorna (bytes, hash_xxh64). Elimina leitura dupla."""
    with open(caminho, "rb") as f:
        dados = f.read()
    return dados, xxhash.xxh64(dados).hexdigest()


def decodificar_imagem(bytes_raw: bytes) -> np.ndarray | None:
    """Decodifica bytes em imagem OpenCV sem tocar em disco.
    Funciona para JPEG/PNG/TIFF etc. NÃO funciona para RAW — use
    decodificar_imagem_para_inferencia() nesse caso.
    """
    arr = np.frombuffer(bytes_raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decodificar_imagem_para_inferencia(
    dados: bytes, caminho: str
) -> np.ndarray | None:
    """Decodifica imagem para inferência (InsightFace).

    Fluxo otimizado:
    1. Se extensão RAW conhecida → direto para rawpy (pula OpenCV)
    2. Senão, tenta OpenCV (rápido — JPEG/PNG/TIFF etc.)
    3. Se OpenCV falhar, tenta rawpy como último recurso

    O numpy array retornado está em BGR (formato OpenCV/InsightFace).
    """
    # ── RAW: pula tentativa inútil do OpenCV (~2ms por foto) ──────────
    if _is_raw(caminho):
        return _decodificar_raw_para_bgr(caminho, dados)

    # ── Fast path: OpenCV ──────────────────────────────────────────────
    arr = np.frombuffer(dados, dtype=np.uint8)
    imagem = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if imagem is not None:
        return imagem

    # ── Fallback (extensão não reconhecida pode ser RAW) ────────────────
    return _decodificar_raw_para_bgr(caminho, dados)


def _decodificar_raw_para_bgr(
    caminho: str, dados: bytes | None = None
) -> np.ndarray | None:
    """Converte arquivo RAW → numpy array BGR.

    **Fluxo otimizado (2 etapas):**

    1. **Fast path** — extrai a visualização JPEG incorporada no arquivo
       RAW (gerada pela câmera). Essa operação leva ~1–10 ms, pois apenas
       lê um bloco de dados JPEG já existente dentro do RAW, sem qualquer
       processamento de sensor/demosaicing.

    2. **Slow path (fallback)** — se o preview JPEG não estiver disponível
       (formato RAW não suportado, thumbnail bitmap, sem preview embutido),
       realiza a conversão completa via ``rawpy.postprocess()`` com
       meia-resolução (half_size).

    Se ``dados`` for fornecido, usa ``io.BytesIO`` (evita segunda leitura
    de disco / rede). Caso contrário, lê do caminho.

    Retorna ``None`` apenas se rawpy não estiver instalado.
    Levanta ``RuntimeError`` se **ambos** os caminhos falharem.
    """
    try:
        import rawpy
    except ImportError:
        print(
            "  ⚠️  rawpy não instalado. RAW não pode ser processado.\n"
            "    → Instale com: pip install rawpy"
        )
        return None

    # Define a fonte: bytes em memória (evita releitura de disco/rede)
    # ou caminho do arquivo (fallback)
    raw_source = io.BytesIO(dados) if dados else caminho

    try:
        preview_bytes: bytes | None = None

        with _raw_decode_lock:
            with rawpy.imread(raw_source) as raw:
                # ── Fast path: preview JPEG incorporado ──────────────────
                # Muitas câmeras embutem uma visualização JPEG (às vezes
                # em resolução total) dentro do RAW. Extrair esse JPEG é
                # muito mais rápido que processar os dados do sensor.
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbnailFormat.JPEG:
                        preview_bytes = thumb.data
                except Exception:
                    pass  # Qualquer erro → fallback seguro

                # ── Slow path (fallback) ─────────────────────────────────
                if preview_bytes is None:
                    par = rawpy.Params()
                    par.use_camera_wb = True
                    par.output_bps = 8
                    par.half_size = RAW_HALF_SIZE
                    par.no_auto_bright = True
                    par.user_qual = 0
                    par.med_passes = 0
                    rgb = raw.postprocess(params=par)

        # Fora do lock: operações OpenCV (liberam o GIL)
        if preview_bytes is not None:
            arr = np.frombuffer(preview_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    except Exception as exc:
        raise RuntimeError(
            f"Falha na conversão RAW: {caminho}\n"
            f"  Motivo: {exc}"
        ) from exc


def redimensionar_se_necessario(
    imagem: np.ndarray, max_dim: int = MAX_DIMENSION
) -> np.ndarray:
    """Redimensiona mantendo proporção se largura ou altura > max_dim."""
    altura, largura = imagem.shape[:2]
    if max(largura, altura) <= max_dim:
        return imagem
    fator = max_dim / max(largura, altura)
    novo_tamanho = (int(largura * fator), int(altura * fator))
    return cv2.resize(imagem, novo_tamanho, interpolation=cv2.INTER_AREA)
