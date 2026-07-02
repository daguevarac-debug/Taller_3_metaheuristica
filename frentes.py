"""Generación de los frentes de Pareto consolidados."""

from pathlib import Path

import numpy as np
import pandas as pd

from analisis import extraer_frente_no_dominado
from main import (
    COEFICIENTE_COGNITIVO,
    COEFICIENTE_SOCIAL,
    INERCIA_FINAL,
    INERCIA_INICIAL,
    ITERACIONES,
    PROBABILIDAD_CRUCE,
    PROBABILIDAD_MUTACION,
    SEMILLAS,
    TAMANO_POBLACION,
    TAMANO_REPOSITORIO,
)
from mopso import ejecutar_mopso
from nsga2 import ejecutar_nsga2


def construir_tabla_frente(
    soluciones: np.ndarray,
    objetivos: np.ndarray,
) -> pd.DataFrame:
    """Organiza un frente mediante variables, ganancia y tiempo."""
    return pd.DataFrame(
        {
            "x1": soluciones[:, 0].astype(int),
            "x2": soluciones[:, 1].astype(int),
            "x3": soluciones[:, 2].astype(int),
            "ganancia": -objetivos[:, 0],
            "tiempo": objetivos[:, 1],
        }
    )


def consolidar_frente_desde_soluciones(
    soluciones: list[np.ndarray] | np.ndarray,
) -> pd.DataFrame:
    """Consolida candidatos y devuelve su frente no dominado."""
    if isinstance(soluciones, list):
        soluciones_candidatas = np.vstack(soluciones)
    else:
        soluciones_candidatas = np.asarray(
            soluciones,
            dtype=int,
        )

    frente, objetivos = extraer_frente_no_dominado(
        soluciones_candidatas,
        eliminar_duplicados=True,
    )

    return construir_tabla_frente(
        frente,
        objetivos,
    )


def consolidar_frentes_desde_corridas(
    resultados_corridas: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolida frentes de NSGA-II y repositorios de MOPSO ya calculados."""
    candidatos_nsga2 = [
        corrida["frente_nsga2"]
        for corrida in resultados_corridas
    ]

    candidatos_mopso = [
        corrida["repositorio_mopso"]
        for corrida in resultados_corridas
    ]

    return (
        consolidar_frente_desde_soluciones(
            candidatos_nsga2
        ),
        consolidar_frente_desde_soluciones(
            candidatos_mopso
        ),
    )


def generar_frentes_consolidados() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Consolida las soluciones no dominadas de las cinco corridas."""
    candidatos_nsga2 = []
    candidatos_mopso = []

    for numero_corrida, semilla in enumerate(
        SEMILLAS,
        start=1,
    ):
        print(
            f"Corrida {numero_corrida} de {len(SEMILLAS)} "
            f"- semilla {semilla}"
        )

        _, frente_nsga2, _ = ejecutar_nsga2(
            tamano_poblacion=TAMANO_POBLACION,
            iteraciones=ITERACIONES,
            probabilidad_cruce=PROBABILIDAD_CRUCE,
            probabilidad_mutacion=PROBABILIDAD_MUTACION,
            semilla=semilla,
        )

        _, repositorio_mopso, _ = ejecutar_mopso(
            tamano_enjambre=TAMANO_POBLACION,
            iteraciones=ITERACIONES,
            inercia_inicial=INERCIA_INICIAL,
            inercia_final=INERCIA_FINAL,
            coeficiente_cognitivo=COEFICIENTE_COGNITIVO,
            coeficiente_social=COEFICIENTE_SOCIAL,
            tamano_repositorio=TAMANO_REPOSITORIO,
            semilla=semilla,
        )

        candidatos_nsga2.append(frente_nsga2)
        candidatos_mopso.append(repositorio_mopso)

    soluciones_nsga2 = np.vstack(candidatos_nsga2)
    soluciones_mopso = np.vstack(candidatos_mopso)

    frente_nsga2, objetivos_nsga2 = (
        extraer_frente_no_dominado(
            soluciones_nsga2,
            eliminar_duplicados=True,
        )
    )

    frente_mopso, objetivos_mopso = (
        extraer_frente_no_dominado(
            soluciones_mopso,
            eliminar_duplicados=True,
        )
    )

    return (
        construir_tabla_frente(
            frente_nsga2,
            objetivos_nsga2,
        ),
        construir_tabla_frente(
            frente_mopso,
            objetivos_mopso,
        ),
    )


if __name__ == "__main__":
    carpeta_resultados = Path("resultados")
    carpeta_resultados.mkdir(
        parents=True,
        exist_ok=True,
    )

    frente_nsga2, frente_mopso = (
        generar_frentes_consolidados()
    )

    ruta_nsga2 = (
        carpeta_resultados
        / "frente_pareto_nsga2.csv"
    )

    ruta_mopso = (
        carpeta_resultados
        / "frente_pareto_mopso.csv"
    )

    frente_nsga2.to_csv(
        ruta_nsga2,
        index=False,
        encoding="utf-8-sig",
    )

    frente_mopso.to_csv(
        ruta_mopso,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nFrente consolidado de NSGA-II:",
        len(frente_nsga2),
        "soluciones únicas.",
    )

    print(
        "Frente consolidado de MOPSO:",
        len(frente_mopso),
        "soluciones únicas.",
    )

    print("\nArchivos guardados en:")
    print(ruta_nsga2)
    print(ruta_mopso)
