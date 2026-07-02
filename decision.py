"""Construcción de la matriz conjunta para la toma de decisión."""

from pathlib import Path

import numpy as np
import pandas as pd

from analisis import mascara_no_dominados


COLUMNAS_ALTERNATIVA = [
    "x1",
    "x2",
    "x3",
    "ganancia",
    "tiempo",
]


def combinar_metodos(valores: pd.Series) -> str:
    """Combina la procedencia de alternativas equivalentes."""
    metodos = set()

    for valor in valores:
        if valor == "Ambos":
            metodos.update(
                {
                    "NSGA-II",
                    "MOPSO",
                }
            )
        else:
            metodos.add(valor)

    if metodos == {"NSGA-II", "MOPSO"}:
        return "Ambos"

    return next(iter(metodos))


def leer_frente(
    ruta: Path,
    metodo: str,
) -> pd.DataFrame:
    """Lee y valida un frente de Pareto almacenado en CSV."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    tabla = pd.read_csv(
        ruta,
        encoding="utf-8-sig",
    )

    faltantes = [
        columna
        for columna in COLUMNAS_ALTERNATIVA
        if columna not in tabla.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en {ruta.name}: {faltantes}"
        )

    tabla = tabla[COLUMNAS_ALTERNATIVA].copy()

    tabla[["x1", "x2", "x3"]] = (
        tabla[["x1", "x2", "x3"]]
        .astype(int)
    )

    tabla[["ganancia", "tiempo"]] = (
        tabla[["ganancia", "tiempo"]]
        .astype(float)
    )

    tabla["metodo"] = metodo

    return tabla


def construir_matriz_decision(
    carpeta_resultados: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Combina ambos frentes y conserva solamente alternativas
    no dominadas respecto al conjunto completo.
    """
    ruta_nsga2 = (
        carpeta_resultados
        / "frente_pareto_nsga2.csv"
    )

    ruta_mopso = (
        carpeta_resultados
        / "frente_pareto_mopso.csv"
    )

    frente_nsga2 = leer_frente(
        ruta_nsga2,
        "NSGA-II",
    )

    frente_mopso = leer_frente(
        ruta_mopso,
        "MOPSO",
    )

    tabla_completa = pd.concat(
        [
            frente_nsga2,
            frente_mopso,
        ],
        ignore_index=True,
    )

    # Se agrupan soluciones exactamente iguales y se registra
    # si fueron encontradas por uno o por ambos algoritmos.
    alternativas_unicas = (
        tabla_completa
        .groupby(
            COLUMNAS_ALTERNATIVA,
            as_index=False,
        )
        .agg(
            metodo=(
                "metodo",
                combinar_metodos,
            )
        )
    )

    # La dominancia se evalúa con la formulación interna:
    # minimizar -ganancia y minimizar tiempo.
    objetivos = np.column_stack(
        (
            -alternativas_unicas[
                "ganancia"
            ].to_numpy(),
            alternativas_unicas[
                "tiempo"
            ].to_numpy(),
        )
    )

    mascara = mascara_no_dominados(
        objetivos
    )

    alternativas_no_dominadas = (
        alternativas_unicas.loc[mascara]
        .copy()
    )

    # Dos vectores diferentes pueden producir exactamente la
    # misma ganancia y el mismo tiempo. Se conserva una
    # alternativa representativa y se registra cuántas existen.
    filas_matriz = []

    grupos_objetivos = (
        alternativas_no_dominadas
        .groupby(
            [
                "ganancia",
                "tiempo",
            ],
            sort=True,
        )
    )

    for _, grupo in grupos_objetivos:
        grupo_ordenado = grupo.sort_values(
            [
                "x1",
                "x2",
                "x3",
                "metodo",
            ],
            kind="mergesort",
        )

        representante = (
            grupo_ordenado
            .iloc[0]
            .to_dict()
        )

        representante["metodo"] = (
            combinar_metodos(
                grupo["metodo"]
            )
        )

        representante[
            "alternativas_equivalentes"
        ] = len(grupo)

        filas_matriz.append(
            representante
        )

    matriz_decision = pd.DataFrame(
        filas_matriz
    )

    matriz_decision = (
        matriz_decision
        .sort_values(
            [
                "tiempo",
                "ganancia",
                "x1",
                "x2",
                "x3",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    matriz_decision.insert(
        0,
        "alternativa",
        np.arange(
            1,
            len(matriz_decision) + 1,
        ),
    )

    matriz_decision[
        [
            "x1",
            "x2",
            "x3",
            "alternativas_equivalentes",
        ]
    ] = matriz_decision[
        [
            "x1",
            "x2",
            "x3",
            "alternativas_equivalentes",
        ]
    ].astype(int)

    estadisticas = {
        "soluciones_nsga2": len(
            frente_nsga2
        ),
        "soluciones_mopso": len(
            frente_mopso
        ),
        "soluciones_totales": len(
            tabla_completa
        ),
        "alternativas_unicas": len(
            alternativas_unicas
        ),
        "alternativas_dominadas": (
            len(alternativas_unicas)
            - len(alternativas_no_dominadas)
        ),
        "alternativas_no_dominadas": len(
            alternativas_no_dominadas
        ),
        "pares_objetivo_unicos": len(
            matriz_decision
        ),
    }

    return (
        matriz_decision,
        estadisticas,
    )


if __name__ == "__main__":
    carpeta_resultados = Path(
        "resultados"
    )

    matriz, estadisticas = (
        construir_matriz_decision(
            carpeta_resultados
        )
    )

    ruta_salida = (
        carpeta_resultados
        / "matriz_decision.csv"
    )

    matriz.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nConstrucción de la matriz de decisión"
    )

    for nombre, valor in estadisticas.items():
        print(
            f"{nombre}: {valor}"
        )

    print(
        "\nPrimeras alternativas"
    )

    print(
        matriz.head(10).to_string(
            index=False
        )
    )

    print(
        "\nArchivo guardado en:",
        ruta_salida,
    )