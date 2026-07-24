"""Utilitários de entrada/saída de arquivos."""

import shutil
from pathlib import Path
from typing import List

from config import IMAGE_EXTENSIONS, RAW_EXTENSIONS

# Extensões aceitas no total (imagens comuns + RAW)
_ALL_EXTENSIONS = IMAGE_EXTENSIONS | RAW_EXTENSIONS


def is_raw_file(path: Path) -> bool:
    """Retorna True se o arquivo é um RAW conhecido."""
    return path.suffix.lower() in RAW_EXTENSIONS


def listar_imagens(pasta: str) -> List[Path]:
    """Lista todas as imagens (JPEG, PNG, TIFF, RAW, etc.) recursivamente na pasta."""
    return [
        p
        for p in Path(pasta).rglob("*")
        if p.is_file() and p.suffix.lower() in _ALL_EXTENSIONS
    ]


def pedir_pasta(mensagem: str, deve_existir: bool = True) -> str:
    """Solicita um caminho de pasta ao usuário com validação."""
    while True:
        caminho = Path(input(mensagem).strip().strip('"').strip("'"))
        if deve_existir and not caminho.is_dir():
            print(f"  ❌ Pasta não encontrada: '{caminho}'. Tente novamente.")
            continue
        if not deve_existir:
            caminho.mkdir(parents=True, exist_ok=True)
        return str(caminho)


def transferir_para_subpasta(
    foto: Path, subpasta: str, pasta_destino: str, indice: int,
    mover: bool = False,
) -> None:
    """Move ou copia uma foto para uma subpasta de destino."""
    destino_pasta = Path(pasta_destino) / subpasta
    destino_pasta.mkdir(parents=True, exist_ok=True)
    destino = destino_pasta / foto.name
    if destino.exists():
        destino = destino_pasta / f"{foto.stem}_{indice}{foto.suffix}"
    if mover:
        shutil.move(str(foto), str(destino))
    else:
        shutil.copy2(foto, destino)


def copiar_para_subpasta(
    foto: Path, subpasta: str, pasta_destino: str, indice: int
) -> None:
    """Copia uma foto para uma subpasta de destino (compatibilidade)."""
    transferir_para_subpasta(foto, subpasta, pasta_destino, indice, mover=False)
