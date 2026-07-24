"""Serviço de cache com salvamento incremental."""

import json
from pathlib import Path
from typing import Dict

from config import CACHE_FILE, CACHE_INTERVAL


class CacheService:
    """Gerencia o cache de processamento: carregar, verificar, marcar, salvar."""

    def __init__(self, pasta_destino: str) -> None:
        self._pasta_destino = pasta_destino
        self._cache: Dict[str, str] = {}
        self._dirty = False
        self._contador = 0

    def carregar(self) -> None:
        """Carrega o cache do disco, com migração automática de md5→xxhash."""
        caminho_cache = Path(self._pasta_destino) / CACHE_FILE
        if not caminho_cache.exists():
            return
        try:
            with open(caminho_cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        if any(len(v) == 32 for v in cache.values()):
            print("  🔄 Cache em formato antigo (md5). Migrando para xxhash...")
            return
        self._cache = cache

    def ja_processada(self, caminho: str, hash_val: str) -> bool:
        """Verifica se a foto já foi processada com o mesmo hash."""
        return self._cache.get(caminho) == hash_val

    def marcar_processada(self, caminho: str, hash_val: str) -> None:
        """Marca uma foto como processada com sucesso."""
        self._cache[caminho] = hash_val
        self._dirty = True
        self._contador += 1

    def marcar_erro(self, caminho: str) -> None:
        """Marca uma foto que não pôde ser lida (hash vazio)."""
        self._cache[caminho] = ""
        self._dirty = True
        self._contador += 1

    def salvar_incrementalmente(self) -> None:
        """Salva o cache se o contador atingiu o intervalo configurado."""
        if self._dirty and self._contador >= CACHE_INTERVAL:
            self._salvar()
            self._contador = 0

    def salvar_final(self) -> None:
        """Salva o cache ao final do processamento, se necessário."""
        if self._dirty:
            self._salvar()

    def _salvar(self) -> None:
        caminho_cache = Path(self._pasta_destino) / CACHE_FILE
        with open(caminho_cache, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, separators=(",", ":"))
        self._dirty = False

    def snapshot(self) -> Dict[str, str]:
        """Retorna uma cópia do cache atual (usado pelos workers paralelos)."""
        return dict(self._cache)

    def __len__(self) -> int:
        return len(self._cache)
