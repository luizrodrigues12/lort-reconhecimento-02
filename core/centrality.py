"""Análise de foco facial: ranqueia rostos por centralidade, tamanho e confiança do detector."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ─── Configuração de pesos (ajustável externamente) ────────────────────────

@dataclass
class PesosMetricas:
    """Pesos das métricas de ranqueamento. Altere aqui para recalibrar."""
    centralidade: float = 0.40
    tamanho: float = 0.20
    det_score: float = 0.40


# ─── Função pública e isolada para cálculo do score ────────────────────────

def calcular_score_total(
    centralidade: float,
    tamanho: float,
    det_score: float,
    pesos: PesosMetricas | None = None,
) -> float:
    """Score ponderado final — isolada para testes sem instanciar AnalisadorFoco."""
    p = pesos or PesosMetricas()
    return (
        p.centralidade * centralidade
        + p.tamanho * tamanho
        + p.det_score * det_score
    )


# ─── Métricas individuais (stateless) ──────────────────────────────────────

def _centralidade_da_bbox(bbox: tuple, shape_imagem: tuple) -> float:
    """1.0 no centro da imagem, decai linearmente até 0 nos cantos."""
    altura, largura = shape_imagem[:2]
    x1, y1, x2, y2 = bbox
    cx_r = (x1 + x2) / 2.0
    cy_r = (y1 + y2) / 2.0
    cx_i, cy_i = largura / 2.0, altura / 2.0
    distancia = np.hypot(cx_r - cx_i, cy_r - cy_i)
    return 1.0 - (distancia / np.hypot(largura / 2.0, altura / 2.0))


def _tamanho_relativo(bbox: tuple, shape_imagem: tuple) -> float:
    """Área do rosto dividida pela área total da imagem (cap 1.0)."""
    altura, largura = shape_imagem[:2]
    x1, y1, x2, y2 = bbox
    return min(1.0, (x2 - x1) * (y2 - y1) / (largura * altura))


def _extrair_det_score(rosto) -> float:
    """Confiança do detector InsightFace (atributo nativo, 0–1). Sem custo extra."""
    return float(getattr(rosto, "det_score", 0.0))


# ─── Classe principal ──────────────────────────────────────────────────────

class AnalisadorFoco:
    """Ranqueia rostos e decide qual é o principal em uma foto."""

    def __init__(
        self,
        pesos: PesosMetricas | None = None,
        limiar_dominancia: float = 1.6,
    ) -> None:
        self.pesos = pesos or PesosMetricas()
        self.limiar_dominancia = limiar_dominancia

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analisar_foco(self, rosto, imagem_shape: tuple) -> Dict:
        """Retorna todas as métricas + score final para um rosto."""
        bbox = rosto.bbox
        centralidade = _centralidade_da_bbox(bbox, imagem_shape)
        tamanho = _tamanho_relativo(bbox, imagem_shape)
        det_score = _extrair_det_score(rosto)
        pontuacao = calcular_score_total(
            centralidade, tamanho, det_score, self.pesos
        )
        return {
            "centralidade": centralidade,
            "tamanho_percentual": tamanho,
            "det_score": det_score,
            "pontuacao": pontuacao,
        }

    def encontrar_rosto_dominante(
        self, rostos_com_analise: List[Tuple]
    ) -> Tuple[int, bool, float]:
        """
        Retorna (índice_do_melhor, é_dominante?, confiança).

        *is_dominante* = True se a razão melhor/segundo >= *limiar_dominancia*.
        """
        if not rostos_com_analise:
            return -1, False, 0.0
        if len(rostos_com_analise) == 1:
            return 0, True, 1.0

        melhor, segundo = _top2_pontuacoes(rostos_com_analise)
        pontuacoes = [a["pontuacao"] for _, _, a in rostos_com_analise]
        razao = (
            pontuacoes[melhor] / pontuacoes[segundo]
            if pontuacoes[segundo] > 0
            else float("inf")
        )

        tem_dominante = razao >= self.limiar_dominancia
        confianca = _calcular_confianca(razao, tem_dominante, self.limiar_dominancia)
        return melhor, tem_dominante, confianca


# ─── Helpers internos ──────────────────────────────────────────────────────

def _top2_pontuacoes(rostos_com_analise: List[Tuple]) -> Tuple[int, int]:
    """Índices do maior e segundo maior score (uma passada, O(n))."""
    pontuacoes = [a["pontuacao"] for _, _, a in rostos_com_analise]
    melhor, segundo = (0, 1) if pontuacoes[0] >= pontuacoes[1] else (1, 0)
    for i in range(2, len(pontuacoes)):
        if pontuacoes[i] > pontuacoes[melhor]:
            segundo = melhor
            melhor = i
        elif pontuacoes[i] > pontuacoes[segundo]:
            segundo = i
    return melhor, segundo


def _calcular_confianca(razao: float, tem_dominante: bool, limiar: float) -> float:
    """Mapeia a razão melhor/segundo para um valor de confiança entre 0 e 1."""
    confianca = min(1.0, razao / limiar * 0.8 + 0.2)
    if tem_dominante:
        return max(0.5, confianca)
    return (1.0 - confianca) / 2
