"""Aplicación de TOPSIS y suma ponderada normalizada."""

from pathlib import Path

import numpy as np
import pandas as pd


PESO_GANANCIA = 0.50
PESO_TIEMPO = 0.50
EPSILON = 1e-12

COLUMNAS_REQUERIDAS = [
    "alternativa",
    "x1",
    "x2",
    "x3",
    "ganancia",
    "tiempo",
    "metodo",
    "alternativas_equivalentes",
]


def leer_matriz_decision(
    ruta: Path,
) -> pd.DataFrame:
    """Lee y valida la matriz de alternativas."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    tabla = pd.read_csv(
        ruta,
        encoding="utf-8-sig",
    )

    columnas_faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in tabla.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "La matriz de decisión no contiene las columnas: "
            f"{columnas_faltantes}"
        )

    if tabla.empty:
        raise ValueError(
            "La matriz de decisión está vacía."
        )

    tabla = tabla[COLUMNAS_REQUERIDAS].copy()

    tabla[
        [
            "alternativa",
            "x1",
            "x2",
            "x3",
            "alternativas_equivalentes",
        ]
    ] = tabla[
        [
            "alternativa",
            "x1",
            "x2",
            "x3",
            "alternativas_equivalentes",
        ]
    ].astype(int)

    tabla[
        [
            "ganancia",
            "tiempo",
        ]
    ] = tabla[
        [
            "ganancia",
            "tiempo",
        ]
    ].astype(float)

    return tabla


def normalizar_criterios(
    tabla: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza la ganancia como beneficio y el tiempo como costo.

    Un valor normalizado alto representa un mejor desempeño.
    """
    resultado = tabla.copy()

    ganancia = resultado[
        "ganancia"
    ].to_numpy(dtype=float)

    tiempo = resultado[
        "tiempo"
    ].to_numpy(dtype=float)

    rango_ganancia = (
        np.max(ganancia)
        - np.min(ganancia)
    )

    rango_tiempo = (
        np.max(tiempo)
        - np.min(tiempo)
    )

    if np.isclose(rango_ganancia, 0.0):
        raise ValueError(
            "Todas las alternativas tienen la misma ganancia."
        )

    if np.isclose(rango_tiempo, 0.0):
        raise ValueError(
            "Todas las alternativas tienen el mismo tiempo."
        )

    # Criterio de beneficio: un valor mayor es mejor.
    resultado[
        "ganancia_normalizada"
    ] = (
        ganancia - np.min(ganancia)
    ) / rango_ganancia

    # Criterio de costo: un valor menor es mejor.
    resultado[
        "tiempo_normalizado"
    ] = (
        np.max(tiempo) - tiempo
    ) / rango_tiempo

    return resultado


def aplicar_topsis(
    tabla: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """Calcula el coeficiente de cercanía de TOPSIS."""
    resultado = tabla.copy()

    matriz_normalizada = resultado[
        [
            "ganancia_normalizada",
            "tiempo_normalizado",
        ]
    ].to_numpy(dtype=float)

    pesos = np.array(
        [
            PESO_GANANCIA,
            PESO_TIEMPO,
        ],
        dtype=float,
    )

    matriz_ponderada = (
        matriz_normalizada * pesos
    )

    ideal_positivo = np.max(
        matriz_ponderada,
        axis=0,
    )

    ideal_negativo = np.min(
        matriz_ponderada,
        axis=0,
    )

    distancia_positiva = np.linalg.norm(
        matriz_ponderada - ideal_positivo,
        axis=1,
    )

    distancia_negativa = np.linalg.norm(
        matriz_ponderada - ideal_negativo,
        axis=1,
    )

    coeficiente_cercania = (
        distancia_negativa
        / (
            distancia_positiva
            + distancia_negativa
            + EPSILON
        )
    )

    resultado[
        "distancia_ideal_positiva"
    ] = distancia_positiva

    resultado[
        "distancia_ideal_negativa"
    ] = distancia_negativa

    resultado[
        "puntaje_topsis"
    ] = coeficiente_cercania

    resultado[
        "ranking_topsis"
    ] = (
        resultado["puntaje_topsis"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return (
        resultado,
        ideal_positivo,
        ideal_negativo,
    )


def aplicar_suma_ponderada(
    tabla: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula el puntaje de suma ponderada normalizada."""
    resultado = tabla.copy()

    resultado[
        "puntaje_suma_ponderada"
    ] = (
        PESO_GANANCIA
        * resultado[
            "ganancia_normalizada"
        ]
        + PESO_TIEMPO
        * resultado[
            "tiempo_normalizado"
        ]
    )

    resultado[
        "ranking_suma_ponderada"
    ] = (
        resultado[
            "puntaje_suma_ponderada"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return resultado


def construir_tabla_seleccionadas(
    tabla: pd.DataFrame,
) -> pd.DataFrame:
    """Extrae la mejor alternativa de cada técnica."""
    indice_topsis = tabla[
        "puntaje_topsis"
    ].idxmax()

    indice_suma = tabla[
        "puntaje_suma_ponderada"
    ].idxmax()

    solucion_topsis = tabla.loc[
        indice_topsis
    ]

    solucion_suma = tabla.loc[
        indice_suma
    ]

    columnas_salida = [
        "alternativa",
        "x1",
        "x2",
        "x3",
        "ganancia",
        "tiempo",
        "metodo",
    ]

    fila_topsis = {
        "tecnica": "TOPSIS",
        **solucion_topsis[
            columnas_salida
        ].to_dict(),
        "puntaje": float(
            solucion_topsis[
                "puntaje_topsis"
            ]
        ),
    }

    fila_suma = {
        "tecnica": "Suma ponderada normalizada",
        **solucion_suma[
            columnas_salida
        ].to_dict(),
        "puntaje": float(
            solucion_suma[
                "puntaje_suma_ponderada"
            ]
        ),
    }

    seleccionadas = pd.DataFrame(
        [
            fila_topsis,
            fila_suma,
        ]
    )

    seleccionadas[
        [
            "alternativa",
            "x1",
            "x2",
            "x3",
        ]
    ] = seleccionadas[
        [
            "alternativa",
            "x1",
            "x2",
            "x3",
        ]
    ].astype(int)

    return seleccionadas


if __name__ == "__main__":
    carpeta_resultados = Path(
        "resultados"
    )

    ruta_matriz = (
        carpeta_resultados
        / "matriz_decision.csv"
    )

    tabla = leer_matriz_decision(
        ruta_matriz
    )

    tabla = normalizar_criterios(
        tabla
    )

    (
        tabla,
        ideal_positivo,
        ideal_negativo,
    ) = aplicar_topsis(
        tabla
    )

    tabla = aplicar_suma_ponderada(
        tabla
    )

    seleccionadas = (
        construir_tabla_seleccionadas(
            tabla
        )
    )

    ruta_ranking = (
        carpeta_resultados
        / "ranking_multicriterio.csv"
    )

    ruta_seleccionadas = (
        carpeta_resultados
        / "soluciones_multicriterio.csv"
    )

    tabla.to_csv(
        ruta_ranking,
        index=False,
        encoding="utf-8-sig",
    )

    seleccionadas.to_csv(
        ruta_seleccionadas,
        index=False,
        encoding="utf-8-sig",
    )

    columnas_impresion_topsis = [
        "ranking_topsis",
        "alternativa",
        "x1",
        "x2",
        "x3",
        "ganancia",
        "tiempo",
        "metodo",
        "puntaje_topsis",
    ]

    columnas_impresion_suma = [
        "ranking_suma_ponderada",
        "alternativa",
        "x1",
        "x2",
        "x3",
        "ganancia",
        "tiempo",
        "metodo",
        "puntaje_suma_ponderada",
    ]

    mejores_topsis = (
        tabla
        .sort_values(
            "puntaje_topsis",
            ascending=False,
        )
        .head(5)
    )

    mejores_suma = (
        tabla
        .sort_values(
            "puntaje_suma_ponderada",
            ascending=False,
        )
        .head(5)
    )

    print(
        "\nPesos utilizados"
    )

    print(
        f"Ganancia: {PESO_GANANCIA:.2f}"
    )

    print(
        f"Tiempo: {PESO_TIEMPO:.2f}"
    )

    print(
        "\nSolución ideal positiva ponderada:",
        ideal_positivo,
    )

    print(
        "Solución ideal negativa ponderada:",
        ideal_negativo,
    )

    print(
        "\nCinco mejores alternativas según TOPSIS"
    )

    print(
        mejores_topsis[
            columnas_impresion_topsis
        ].to_string(
            index=False
        )
    )

    print(
        "\nCinco mejores alternativas según "
        "suma ponderada normalizada"
    )

    print(
        mejores_suma[
            columnas_impresion_suma
        ].to_string(
            index=False
        )
    )

    print(
        "\nSoluciones seleccionadas"
    )

    print(
        seleccionadas.to_string(
            index=False
        )
    )

    print(
        "\nArchivos guardados en:"
    )

    print(ruta_ranking)
    print(ruta_seleccionadas)