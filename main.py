"""
Organizador de Fotos por Pessoa usando Reconhecimento Facial
COM PARALELIZAÇÃO CPU E ENCERRAMENTO CONTROLADO

Melhorias desta versão:
  • ThreadPoolExecutor com sliding window → ~100% de uso da CPU
  • CTRL+C com confirmação → encerramento limpo sem tracebacks
  • Cache compartilhado entre workers
  • RAW decode serializado (lock) → elimina GIL contention
  • Fallback serial: para poucas fotos (< PARALLEL_MIN_PHOTOS)
"""

import os
import shutil
import sys
import time
import warnings
from pathlib import Path

from config import IMAGE_EXTENSIONS, RAW_EXTENSIONS
from core.parallel_processor import ParallelProcessor, ShutdownManager
from services.cache_service import CacheService
from services.face_engine import FaceEngine
from utils.file_utils import listar_imagens, pedir_pasta
from utils.time_utils import formatar_tempo

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── Helpers de terminal ────────────────────────────────────────────────

def print_centralizado(texto: str, largura: int = 60) -> None:
    print(texto.center(largura))


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Instala handler de CTRL+C / SIGTERM ─────────────────────────
    shutdown = ShutdownManager.instancia()
    shutdown.instalar()

    # ── Tenta capturar SystemExit do signal handler para sair limpo ─
    #    (sem traceback, apenas mensagem amigável)
    try:

        _executar_pipeline()

    except SystemExit:
        # Signal handler com confirmação já tratou a saída
        sys.exit(0)
    except KeyboardInterrupt:
        # Caso raro de CTRL+C não capturado pelo ShutdownManager
        print("\n\n👋 Processamento interrompido pelo usuário.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Erro inesperado: {exc}")
        sys.exit(1)


def _executar_pipeline() -> None:
    os.system("cls" if os.name == "nt" else "clear")

    tempo_inicio = time.time()

    print("=" * 60)
    print_centralizado("ORGANIZADOR DE FOTOS POR PESSOA")
    print("=" * 60)
    print("\nEste programa identifica quem é o foco da foto baseado em:\n")
    print("- Centralidade (quão perto do centro está)")
    print("- Tamanho do rosto (proporção na imagem)")
    print("- Confiança do detector (det_score do InsightFace)\n")

    # ── Pastas ──────────────────────────────────────────────────────
    pasta_fotos = pedir_pasta(
        "> Pasta com TODAS as fotos: ", deve_existir=True
    )
    pasta_rostos = pedir_pasta(
        "> Pasta com os ROSTOS de referência: ", deve_existir=True
    )
    pasta_destino = pedir_pasta(
        "> Pasta de DESTINO: ", deve_existir=False
    )

    # ── Preferências ────────────────────────────────────────────────
    mover = False
    while True:
        op = input(
            "\nMover ou copiar as fotos? [M/C]: "
        ).strip().upper()
        if op in ("M", "MOVER"):
            mover = True; break
        if op in ("C", "COPIAR"):
            mover = False; break
        print("  Resposta inválida. Digite M (mover) ou C (copiar).")

    separar = True
    while True:
        op = input(
            "Separar fotos de GRUPOS? [S/N]: "
        ).strip().upper()
        if op in ("S", "SIM"):
            separar = True; break
        if op in ("N", "NAO", "NÃO"):
            separar = False; break
        print("  Resposta inválida. Digite S (sim) ou N (não).")

    criar_nao_reconhecidos = True
    while True:
        op = input(
            "\nCriar pasta 'Não Reconhecidos' para fotos sem rosto? [S/N]: "
        ).strip().upper()
        if op in ("S", "SIM"):
            criar_nao_reconhecidos = True; break
        if op in ("N", "NAO", "NÃO"):
            criar_nao_reconhecidos = False; break
        print("  Resposta inválida. Digite S (sim) ou N (não).")

    # ── Cache ───────────────────────────────────────────────────────
    cache_service = CacheService(pasta_destino)
    cache_service.carregar()
    if len(cache_service) > 0:
        print(
            f"\nCache carregado: {len(cache_service)} foto(s) já processadas."
        )

    # ── Face Engine (CPU) ───────────────────────────────────────────
    face_engine = FaceEngine()

    # ── Referências ─────────────────────────────────────────────────
    print("─" * 60)
    referencias, matriz_refs, nomes_refs = face_engine.carregar_referencias(
        pasta_rostos
    )
    print(
        f"\n- {len(referencias)} ROSTO(S) mapeado(s): "
        f"{', '.join(nomes_refs)} \n"
    )

    # ── Fotos ───────────────────────────────────────────────────────
    fotos = listar_imagens(pasta_fotos)
    if not fotos:
        print("⚠️ Nenhuma imagem encontrada.")
        sys.exit(0)

    # ── Verificação de espaço ───────────────────────────────────────
    _verificar_espaco(fotos, pasta_destino, mover)

    # ── Processamento ──────────────────────────────────────────────
    processador = ParallelProcessor(
        face_engine=face_engine,
        cache_service=cache_service,
        pasta_destino=pasta_destino,
        mover=mover,
        separar_grupos=separar,
        criar_pasta_nao_reconhecidos=criar_nao_reconhecidos,
    )
    resultado = processador.processar(fotos, matriz_refs, nomes_refs)

    # ── Resumo final ───────────────────────────────────────────────
    tempo_total = time.time() - tempo_inicio
    print("\n" + "=" * 60)
    print_centralizado("RESUMO FINAL")
    print("=" * 60)

    filtrados = [
        (nome, qtd)
        for nome, qtd in resultado.get("contadores", {}).items()
        if qtd > 0
    ]

    if filtrados:
        print("\nFotos organizadas por pessoa:\n")
        for nome, qtd in sorted(
            resultado.get("contadores", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if qtd > 0:
                print(f" - {nome:<30} {qtd:>4} foto(s)")
    else:
        print("\nATENÇÃO: Nenhum rosto reconhecido foi identificado!")

    print("\nCategorias:\n")
    print(
        " - Fotos com múltiplas pessoas (grupos): "
        f"{resultado.get('total_grupos', 0):>4}"
    )
    print(
        " - Fotos não reconhecidas:               "
        f"{resultado.get('total_nao_reconhecidos', 0):>4}"
    )
    print(
        " - Fotos em cache (ignoradas):           "
        f"{resultado.get('total_cache', 0):>4}"
    )
    print(
        " - Total de rostos detectados:           "
        f"{resultado.get('total_rostos', 0):>4}"
    )
    print(
        f"\nTempo total de processamento: "
        f"{formatar_tempo(tempo_total):>4}"
    )
    print(f"\nDestino: {pasta_destino} \n")
    print("=" * 60)
    print(
        "\nConcluído! A decisão foi baseada em centralidade,"
        " tamanho e det_score.\n"
    )

    # ── Relatório ───────────────────────────────────────────────────
    _gerar_relatorio(pasta_destino, len(fotos), resultado)

    try:
        input("\nPressione Enter para sair...")
    except EOFError:
        pass


def _verificar_espaco(
    fotos: list, pasta_destino: str, mover: bool
) -> None:
    """Verifica se há espaço livre suficiente no destino e alerta se não houver."""
    total_bytes = sum(f.stat().st_size for f in fotos)
    try:
        uso = shutil.disk_usage(pasta_destino)
        livre = uso.free
    except OSError:
        return

    if livre >= total_bytes * 1.1:
        return

    gb_necessario = total_bytes / (1024 ** 3)
    gb_livre = livre / (1024 ** 3)

    print(f"\n⚠️  ESPAÇO INSUFICIENTE NO DESTINO!")
    print(f"   Espaço necessário para as fotos: ~{gb_necessario:.1f} GB")
    print(f"   Espaço livre no destino         : {gb_livre:.1f} GB")

    if mover:
        print("   (Como o modo Mover foi selecionado, menos espaço pode ser")
        print("    suficiente se origem e destino estiverem no mesmo disco.)\n")
    else:
        print("   As fotos serão COPIADAS — o espaço ocupado dobra se origem e")
        print("   destino estiverem no mesmo disco.\n")

    while True:
        op = input("   Continuar mesmo assim? [S/N]: ").strip().upper()
        if op in ("S", "SIM"):
            break
        if op in ("N", "NAO", "NÃO"):
            print("   Processamento cancelado pelo usuário.")
            sys.exit(0)
        print("   Resposta inválida. Digite S (sim) ou N (não).")


def _gerar_relatorio(pasta_destino: str, total_antes: int, resultado: dict) -> None:
    """Gera o arquivo informações-finais.txt na pasta de destino."""
    caminho = Path(pasta_destino) / "informações-finais.txt"

    subpastas: dict[str, int] = {}
    _extensoes = IMAGE_EXTENSIONS | RAW_EXTENSIONS
    for entry in Path(pasta_destino).iterdir():
        if not entry.is_dir():
            continue
        qtd = sum(
            1 for _ in entry.rglob("*")
            if _.is_file() and _.suffix.lower() in _extensoes
        )
        if qtd > 0:
            subpastas[entry.name] = qtd

    total_depois = sum(subpastas.values())

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO FINAL — ORGANIZADOR DE FOTOS".center(70) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total de fotos antes do processo : {total_antes}\n")
        f.write(f"Total de fotos depois do processo: {total_depois}\n")
        f.write("\n" + "—" * 70 + "\n")
        f.write("Fotos por subpasta (ordem alfabética):\n\n")
        for nome in sorted(subpastas.keys()):
            f.write(f"  {nome:<45} {subpastas[nome]:>5} foto(s)\n")

        f.write("\n" + "—" * 70 + "\n")
        f.write("Resumo do processamento:\n\n")
        for nome, qtd in sorted(
            resultado.get("contadores", {}).items(),
            key=lambda x: x[1], reverse=True,
        ):
            if qtd > 0:
                f.write(f"  {nome:<45} {qtd:>5} foto(s)\n")
        f.write(f"  {'Grupos':<45} {resultado.get('total_grupos', 0):>5} foto(s)\n")
        f.write(f"  {'Não Reconhecidos':<45} {resultado.get('total_nao_reconhecidos', 0):>5} foto(s)\n")
        f.write(f"  {'Em cache (ignoradas)':<45} {resultado.get('total_cache', 0):>5} foto(s)\n")

    print(f"\n📄 Relatório salvo em: {caminho}")


if __name__ == "__main__":
    main()
