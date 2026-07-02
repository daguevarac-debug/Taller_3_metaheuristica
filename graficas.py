"""Generación de las gráficas de los frentes de Pareto."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


COLUMNAS_REQUERIDAS = [
    "x1",
    "x2",
    "x3",
    "ganancia",
    "tiempo",
]


def leer_frente(ruta_csv: Path) -> pd.DataFrame:
    """Lee un frente de Pareto desde CSV y valida sus columnas."""
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta_csv}"
        )

    tabla = pd.read_csv(
        ruta_csv,
        encoding="utf-8-sig",
    )

    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in tabla.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en {ruta_csv.name}: {faltantes}"
        )

    tabla = tabla[COLUMNAS_REQUERIDAS].copy()

    tabla[["x1", "x2", "x3"]] = (
        tabla[["x1", "x2", "x3"]]
        .astype(int)
    )

    tabla[["ganancia", "tiempo"]] = (
        tabla[["ganancia", "tiempo"]]
        .astype(float)
    )

    return tabla


def guardar_grafica_frente(
    tabla: pd.DataFrame,
    titulo: str,
    ruta_salida: Path,
) -> None:
    """Guarda una gráfica de dispersión ganancia vs tiempo."""
    tabla_ordenada = tabla.sort_values(
        by=["tiempo", "ganancia"],
        ascending=[True, True],
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        tabla_ordenada["tiempo"],
        tabla_ordenada["ganancia"],
    )
    plt.plot(
        tabla_ordenada["tiempo"],
        tabla_ordenada["ganancia"],
    )

    plt.title(titulo)
    plt.xlabel("Tiempo total de producción [h]")
    plt.ylabel("Ganancia total [$]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    carpeta_resultados = Path("resultados")

    ruta_nsga2_csv = (
        carpeta_resultados
        / "frente_pareto_nsga2.csv"
    )

    ruta_mopso_csv = (
        carpeta_resultados
        / "frente_pareto_mopso.csv"
    )

    ruta_nsga2_png = (
        carpeta_resultados
        / "frente_pareto_nsga2.png"
    )

    ruta_mopso_png = (
        carpeta_resultados
        / "frente_pareto_mopso.png"
    )

    frente_nsga2 = leer_frente(
        ruta_nsga2_csv
    )

    frente_mopso = leer_frente(
        ruta_mopso_csv
    )

    guardar_grafica_frente(
        frente_nsga2,
        "Frente de Pareto consolidado - NSGA-II",
        ruta_nsga2_png,
    )

    guardar_grafica_frente(
        frente_mopso,
        "Frente de Pareto consolidado - MOPSO",
        ruta_mopso_png,
    )

    print("Gráficas generadas correctamente:")
    print(ruta_nsga2_png)
    print(ruta_mopso_png)