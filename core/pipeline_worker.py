"""Processamento individual de uma foto — função pura sem estado global.

Retorna um dicionário com o resultado; a exibição no terminal é feita
pelo chamador (``ParallelProcessor``), que garante ordem sequencial
mesmo quando as fotos são processadas em paralelo.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from core.centrality import AnalisadorFoco
from core.matcher import match_rostos
from services.image_service import (
    decodificar_imagem_para_inferencia,
    ler_bytes_e_hash,
    redimensionar_se_necessario,
)
from utils.file_utils import transferir_para_subpasta, is_raw_file


def processar_unidade(
    foto: Path,
    matriz_refs: np.ndarray,
    nomes_refs: List[str],
    pasta_destino: str,
    app,
    cache_snapshot: Dict[str, str],
    indice: int,
    total: int,
    analisador: Optional["AnalisadorFoco"] = None,
    mover: bool = False,
    separar_grupos: bool = True,
    criar_pasta_nao_reconhecidos: bool = True,
) -> dict:
    """Processa UMA foto e retorna um dicionário com resultado + mensagem.

    **Nunca** chama ``print()`` — toda informação de saída vai no campo
    ``_log_msg`` do dicionário retornado. O chamador decide quando e em
    que ordem exibir as mensagens.

    Campos do retorno
    -----------------
    categoria : str | None
        Nome da pessoa, ``"Grupos"``, ``"Nao Reconhecidos"`` ou ``None``.
    cache : bool
        ``True`` se já estava no cache.
    erro : str | None
        Descrição se houve falha.
    hash : str | None
        Hash xxh64 do arquivo.
    caminho : str
        Caminho normalizado do arquivo.
    qtd_rostos : int
        Rostos detectados na imagem.
    _indice : int
        Número sequencial da foto (para ordenação).
    _log_msg : str
        Linha de progresso formatada (sem quebra de linha final).
    """
    fmt_total = len(str(total))
    prefixo = f"  [{indice:>{fmt_total}}/{total}] {foto.name[:40]:<40}"
    caminho_norm = str(foto.resolve())

    resultado = {
        "categoria": None,
        "cache": False,
        "erro": None,
        "hash": None,
        "caminho": caminho_norm,
        "qtd_rostos": 0,
        "_indice": indice,
        "_log_msg": "",
    }

    # ── 1. Leitura + hash ──────────────────────────────────────────────
    try:
        dados, hash_foto = ler_bytes_e_hash(str(foto))
    except OSError as exc:
        if criar_pasta_nao_reconhecidos:
            transferir_para_subpasta(foto, "Não Reconhecidos", pasta_destino, indice, mover=mover)
        resultado.update({"erro": str(exc), "_log_msg": f"{prefixo} 🚫 erro leitura"})
        return resultado
    resultado["hash"] = hash_foto

    # ── 2. Cache ───────────────────────────────────────────────────────
    if caminho_norm in cache_snapshot and cache_snapshot[caminho_norm] == hash_foto:
        resultado.update({
            "cache": True,
            "_log_msg": f"{prefixo} -> cache (processada anteriormente)",
        })
        return resultado

    # ── 3. Decodificação ───────────────────────────────────────────────
    try:
        imagem = decodificar_imagem_para_inferencia(dados, str(foto))
    except RuntimeError as exc:
        resultado.update({
            "erro": str(exc),
            "_log_msg": f"  ❌ {exc}",
        })
        return resultado

    if imagem is None:
        if is_raw_file(foto):
            resultado.update({
                "erro": "raw_sem_lib",
                "_log_msg": f"{prefixo} ❌ rawpy não instalado",
            })
            return resultado
        if criar_pasta_nao_reconhecidos:
            transferir_para_subpasta(foto, "Não Reconhecidos", pasta_destino, indice, mover=mover)
        resultado.update({
            "erro": "decode",
            "_log_msg": f"{prefixo} 🚫 erro decodificação",
        })
        return resultado

    # ── 4. Detecção facial ─────────────────────────────────────────────
    imagem = redimensionar_se_necessario(imagem)
    rostos = app.get(imagem)

    if not rostos:
        if criar_pasta_nao_reconhecidos:
            transferir_para_subpasta(foto, "Não Reconhecidos", pasta_destino, indice, mover=mover)
        resultado.update({
            "categoria": "Nao Reconhecidos",
            "_log_msg": f"{prefixo} -> Não Reconhecidos (Sem rostos)",
        })
        return resultado

    resultado["qtd_rostos"] = len(rostos)

    # ── 5. Matching ────────────────────────────────────────────────────
    rostos_com_nome = match_rostos(rostos, matriz_refs, nomes_refs)

    if not rostos_com_nome:
        if criar_pasta_nao_reconhecidos:
            transferir_para_subpasta(foto, "Não Reconhecidos", pasta_destino, indice, mover=mover)
        resultado.update({
            "categoria": "Nao Reconhecidos",
            "_log_msg": f"{prefixo} -> Não Reconhecidos [ {len(rostos)} rosto(s) ]",
        })
        return resultado

    # ── 6. Pessoa única ────────────────────────────────────────────────
    nomes_unicos = list({nome for _, nome in rostos_com_nome})
    if len(nomes_unicos) == 1:
        nome = nomes_unicos[0]
        transferir_para_subpasta(foto, nome, pasta_destino, indice, mover=mover)
        resultado.update({
            "categoria": nome,
            "_log_msg": f"{prefixo} -> {nome}",
        })
        return resultado

    # ── 7. Conflito — resolve por foco ─────────────────────────────────
    _analisador = analisador if analisador is not None else AnalisadorFoco()
    analises = [
        (r, n, _analisador.analisar_foco(r, imagem.shape))
        for r, n in rostos_com_nome
    ]
    melhor, dominante, confianca = _analisador.encontrar_rosto_dominante(analises)

    if dominante:
        nome = analises[melhor][1]
        pontuacao = analises[melhor][2]["pontuacao"]
        transferir_para_subpasta(foto, nome, pasta_destino, indice, mover=mover)
        resultado.update({
            "categoria": nome,
            "pontuacao": pontuacao,
            "confianca": confianca,
            "_log_msg": (
                f"{prefixo} -> {nome} "
                f"(score: {pontuacao:.3f}, confianca: {confianca:.0%})"
            ),
        })
    elif separar_grupos:
        nomes_grupo = ", ".join(sorted({n for _, n, _ in analises}))
        transferir_para_subpasta(foto, "Grupos", pasta_destino, indice, mover=mover)
        resultado.update({
            "categoria": "Grupos",
            "nomes_grupo": nomes_grupo,
            "_log_msg": f"{prefixo} -> Grupos ({nomes_grupo})",
        })
    else:
        nome = analises[melhor][1]
        transferir_para_subpasta(foto, nome, pasta_destino, indice, mover=mover)
        resultado.update({
            "categoria": nome,
            "_log_msg": f"{prefixo} -> {nome} (grupo — sem dominância)",
        })

    return resultado
