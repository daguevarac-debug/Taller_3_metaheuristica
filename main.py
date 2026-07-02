"""Ejecución y comparación de cinco corridas de NSGA-II y MOPSO."""

from pathlib import Path
from time import perf_counter

import pandas as pd

from analisis import (
    calcular_spacing,
    evaluar_poblacion,
    extraer_frente_no_dominado,
    porcentaje_no_dominadas,
)
from mopso import ejecutar_mopso
from nsga2 import ejecutar_nsga2


# Condiciones comunes
TAMANO_POBLACION = 30
ITERACIONES = 100
SEMILLAS = [10, 20, 30, 40, 50]

# Parámetros de NSGA-II
PROBABILIDAD_CRUCE = 0.90
PROBABILIDAD_MUTACION = 1.0 / 3.0

# Parámetros de MOPSO
INERCIA_INICIAL = 0.90
INERCIA_FINAL = 0.40
COEFICIENTE_COGNITIVO = 1.50
COEFICIENTE_SOCIAL = 1.50
TAMANO_REPOSITORIO = 100


def calcular_indicadores(
    poblacion_final,
) -> tuple[float, float, int]:
    """
    Calcula los indicadores sobre una población final.

    Devuelve:
    1. Porcentaje de soluciones no dominadas.
    2. Spacing del frente no dominado.
    3. Número de soluciones no dominadas únicas.
    """
    objetivos_poblacion = evaluar_poblacion(
        poblacion_final
    )

    porcentaje = porcentaje_no_dominadas(
        objetivos_poblacion
    )

    (
        soluciones_frente,
        objetivos_frente,
    ) = extraer_frente_no_dominado(
        poblacion_final,
        eliminar_duplicados=True,
    )

    spacing = calcular_spacing(
        objetivos_frente
    )

    return (
        porcentaje,
        spacing,
        len(soluciones_frente),
    )


def ejecutar_corridas() -> pd.DataFrame:
    """Ejecuta cinco corridas independientes de cada método."""
    resultados = []

    for numero_corrida, semilla in enumerate(
        SEMILLAS,
        start=1,
    ):
        print(
            f"\nCorrida {numero_corrida} de {len(SEMILLAS)} "
            f"- semilla {semilla}"
        )

        # NSGA-II
        inicio = perf_counter()

        (
            poblacion_nsga2,
            frente_nsga2,
            objetivos_nsga2,
        ) = ejecutar_nsga2(
            tamano_poblacion=TAMANO_POBLACION,
            iteraciones=ITERACIONES,
            probabilidad_cruce=PROBABILIDAD_CRUCE,
            probabilidad_mutacion=PROBABILIDAD_MUTACION,
            semilla=semilla,
        )

        tiempo_nsga2 = perf_counter() - inicio

        (
            porcentaje_nsga2,
            spacing_nsga2,
            cantidad_nsga2,
        ) = calcular_indicadores(
            poblacion_nsga2
        )

        resultados.append(
            {
                "metodo": "NSGA-II",
                "corrida": numero_corrida,
                "semilla": semilla,
                "soluciones_finales": len(
                    poblacion_nsga2
                ),
                "iteraciones": ITERACIONES,
                "soluciones_no_dominadas": cantidad_nsga2,
                "porcentaje_no_dominadas": porcentaje_nsga2,
                "spacing": spacing_nsga2,
                "tiempo_ejecucion_s": tiempo_nsga2,
            }
        )

        print(
            "NSGA-II terminado. "
            f"No dominadas: {porcentaje_nsga2:.2f} %. "
            f"Spacing: {spacing_nsga2:.6f}. "
            f"Tiempo: {tiempo_nsga2:.3f} s"
        )

        # MOPSO
        inicio = perf_counter()

        (
            enjambre_mopso,
            repositorio_mopso,
            objetivos_repositorio,
        ) = ejecutar_mopso(
            tamano_enjambre=TAMANO_POBLACION,
            iteraciones=ITERACIONES,
            inercia_inicial=INERCIA_INICIAL,
            inercia_final=INERCIA_FINAL,
            coeficiente_cognitivo=COEFICIENTE_COGNITIVO,
            coeficiente_social=COEFICIENTE_SOCIAL,
            tamano_repositorio=TAMANO_REPOSITORIO,
            semilla=semilla,
        )

        tiempo_mopso = perf_counter() - inicio

        (
            porcentaje_mopso,
            spacing_mopso,
            cantidad_mopso,
        ) = calcular_indicadores(
            enjambre_mopso
        )

        resultados.append(
            {
                "metodo": "MOPSO",
                "corrida": numero_corrida,
                "semilla": semilla,
                "soluciones_finales": len(
                    enjambre_mopso
                ),
                "iteraciones": ITERACIONES,
                "soluciones_no_dominadas": cantidad_mopso,
                "porcentaje_no_dominadas": porcentaje_mopso,
                "spacing": spacing_mopso,
                "tiempo_ejecucion_s": tiempo_mopso,
            }
        )

        print(
            "MOPSO terminado. "
            f"No dominadas: {porcentaje_mopso:.2f} %. "
            f"Spacing: {spacing_mopso:.6f}. "
            f"Tiempo: {tiempo_mopso:.3f} s"
        )

    return pd.DataFrame(resultados)


if __name__ == "__main__":
    carpeta_resultados = Path("resultados")
    carpeta_resultados.mkdir(
        parents=True,
        exist_ok=True,
    )

    tabla_corridas = ejecutar_corridas()

    ruta_corridas = (
        carpeta_resultados
        / "indicadores_corridas.csv"
    )

    tabla_corridas.to_csv(
        ruta_corridas,
        index=False,
        encoding="utf-8-sig",
    )

    resumen = (
        tabla_corridas
        .groupby("metodo")
        .agg(
            porcentaje_promedio=(
                "porcentaje_no_dominadas",
                "mean",
            ),
            porcentaje_desviacion=(
                "porcentaje_no_dominadas",
                "std",
            ),
            spacing_promedio=(
                "spacing",
                "mean",
            ),
            spacing_desviacion=(
                "spacing",
                "std",
            ),
            tiempo_promedio_s=(
                "tiempo_ejecucion_s",
                "mean",
            ),
        )
        .reset_index()
    )

    ruta_resumen = (
        carpeta_resultados
        / "comparacion_indicadores.csv"
    )

    resumen.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nResultados por corrida")
    print(
        tabla_corridas.to_string(
            index=False
        )
    )

    print("\nComparación promedio")
    print(
        resumen.to_string(
            index=False
        )
    )

    print(
        "\nArchivos guardados en:",
        ruta_corridas,
        "y",
        ruta_resumen,
    )