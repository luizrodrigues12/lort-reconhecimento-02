"""Módulo de matching vetorizado de embeddings faciais."""

from typing import List, Tuple

import numpy as np

from config import SIMILARITY_THRESHOLD


def match_rostos(
    rostos_detectados: list,
    matriz_refs: np.ndarray,
    nomes_refs: List[str],
    limiar: float = SIMILARITY_THRESHOLD,
) -> List[Tuple]:
    """
    Compara cada rosto detectado com a matriz de referências via dot product.
    Retorna lista de (rosto, nome) para matches acima do limiar.
    """
    rostos_com_nome: List[Tuple] = []
    for rosto in rostos_detectados:
        emb = rosto.normed_embedding
        similaridades: np.ndarray = np.dot(matriz_refs, emb)
        idx_max: int = int(np.argmax(similaridades))
        sim_max: float = float(similaridades[idx_max])
        if sim_max >= limiar:
            rostos_com_nome.append((rosto, nomes_refs[idx_max]))
    return rostos_com_nome
