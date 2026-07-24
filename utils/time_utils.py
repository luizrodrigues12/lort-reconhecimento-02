"""Utilitários de formatação de tempo."""


def formatar_tempo(segundos: float) -> str:
    """Formata segundos em string legível (h m s)."""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    seg = int(segundos % 60)
    if horas > 0:
        return f"{horas}h {minutos}m {seg}s"
    if minutos > 0:
        return f"{minutos}m {seg}s"
    return f"{seg}s"
