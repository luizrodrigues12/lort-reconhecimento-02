"""Serviço de engine de reconhecimento facial (InsightFace) — CPU."""

import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import DET_SIZE, MODEL_NAME


class FaceEngine:
    """Wrapper para o modelo InsightFace (CPU), carregamento lazy."""

    def __init__(self, modo: str = "cpu") -> None:
        self._app: FaceAnalysis | None = None

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            self._inicializar()
        return self._app

    def _inicializar(self) -> None:
        """Carrega o modelo InsightFace com CPUExecutionProvider."""
        print(f"\nCarregando modelo de reconhecimento facial ({MODEL_NAME})...")
        app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        self._app = app
        print("  ✅ Modelo carregado com sucesso!\n")

    # ------------------------------------------------------------------
    # Extração de embeddings
    # ------------------------------------------------------------------

    def extrair_embedding(self, caminho: str) -> np.ndarray | None:
        from services.image_service import decodificar_imagem_para_inferencia

        with open(caminho, "rb") as f:
            dados = f.read()
        imagem = decodificar_imagem_para_inferencia(dados, caminho)
        if imagem is None:
            return None
        rostos = self.app.get(imagem)
        if not rostos:
            return None
        maior = max(
            rostos,
            key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
        )
        return maior.normed_embedding

    # ------------------------------------------------------------------
    # Referências
    # ------------------------------------------------------------------

    def carregar_referencias(
        self, pasta_rostos: str
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
        """Carrega embeddings das fotos de referência e retorna (dict, matriz, nomes)."""
        from utils.file_utils import listar_imagens

        imagens = listar_imagens(pasta_rostos)
        if not imagens:
            print("❌ Nenhuma imagem encontrada na pasta de referencias.")
            sys.exit(1)

        print(f"Encontradas {len(imagens)} foto(s) de referencia. Processando...\n")

        # Força carregamento do modelo agora
        _ = self.app

        referencias: dict[str, np.ndarray] = {}
        for img_path in imagens:
            nome = img_path.stem
            print(f"> Processando '{nome}'...", end=" ")
            embedding = self.extrair_embedding(str(img_path))
            if embedding is None:
                print("sem rosto - ignorado")
                continue
            referencias[nome] = embedding
            print("OK")

        if not referencias:
            print("\nNenhum rosto valido encontrado nas referencias.")
            sys.exit(1)

        nomes = list(referencias.keys())
        matriz = np.array([referencias[n] for n in nomes])
        return referencias, matriz, nomes
