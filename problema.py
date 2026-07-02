"""Definición del problema de producción de escritorios metálicos."""

from __future__ import annotations

import numpy as np


# Límites de las variables:
# x1 = escritorios básicos
# x2 = escritorios estándar
# x3 = escritorios premium
LIMITES_INFERIORES = np.array([0, 0, 0], dtype=int)
LIMITES_SUPERIORES = np.array([60, 50, 40], dtype=int)

# Coeficientes de las funciones objetivo
GANANCIAS = np.array([40.0, 65.0, 90.0])
TIEMPOS = np.array([0.8, 1.2, 1.7])

# Consumo de recursos por tipo de escritorio
# Filas: lámina, corte, soldadura y empaque
# Columnas: básico, estándar y premium
CONSUMOS_RECURSOS = np.array(
    [
        [1.5, 2.0, 2.8],
        [0.30, 0.45, 0.60],
        [0.40, 0.60, 0.90],
        [0.10, 0.15, 0.20],
    ]
)

DISPONIBILIDADES = np.array([180.0, 45.0, 65.0, 18.0])

PRODUCCION_MINIMA = 30


def es_vector_entero(x: np.ndarray) -> bool:
    """Comprueba que una solución tenga tres componentes enteras."""
    x = np.asarray(x)

    return (
        x.shape == (3,)
        and np.all(np.isfinite(x))
        and np.all(x == np.round(x))
    )


def consumo_recursos(x: np.ndarray) -> np.ndarray:
    """Calcula el consumo de lámina, corte, soldadura y empaque."""
    x = np.asarray(x, dtype=float)
    return CONSUMOS_RECURSOS @ x


def es_factible(x: np.ndarray) -> bool:
    """Verifica todas las restricciones del problema."""
    x = np.asarray(x)

    if not es_vector_entero(x):
        return False

    x = x.astype(int)

    cumple_limites = np.all(x >= LIMITES_INFERIORES) and np.all(
        x <= LIMITES_SUPERIORES
    )

    cumple_produccion_minima = np.sum(x) >= PRODUCCION_MINIMA

    cumple_recursos = np.all(
        consumo_recursos(x) <= DISPONIBILIDADES + 1e-10
    )

    return bool(
        cumple_limites
        and cumple_produccion_minima
        and cumple_recursos
    )


def evaluar(x: np.ndarray) -> np.ndarray:
    """
    Evalúa los dos objetivos de minimización.

    f1 = -ganancia
    f2 = tiempo total
    """
    if not es_factible(x):
        raise ValueError("La solución evaluada no es factible.")

    x = np.asarray(x, dtype=int)

    ganancia = float(GANANCIAS @ x)
    tiempo = float(TIEMPOS @ x)

    return np.array([-ganancia, tiempo], dtype=float)


def generar_solucion_factible(
    rng: np.random.Generator,
    max_intentos: int = 10_000,
) -> np.ndarray:
    """Genera aleatoriamente una solución entera factible."""
    for _ in range(max_intentos):
        x = rng.integers(
            low=LIMITES_INFERIORES,
            high=LIMITES_SUPERIORES + 1,
        )

        if es_factible(x):
            return x

    raise RuntimeError(
        "No fue posible generar una solución factible "
        f"después de {max_intentos} intentos."
    )


def generar_poblacion_factible(
    tamano: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una población compuesta solo por soluciones factibles."""
    if tamano <= 0:
        raise ValueError("El tamaño de la población debe ser positivo.")

    poblacion = [
        generar_solucion_factible(rng)
        for _ in range(tamano)
    ]

    return np.array(poblacion, dtype=int)


def resumen_solucion(x: np.ndarray) -> dict[str, float | int]:
    """Devuelve los principales resultados de una solución factible."""
    objetivos = evaluar(x)
    recursos = consumo_recursos(x)

    return {
        "basicos": int(x[0]),
        "estandar": int(x[1]),
        "premium": int(x[2]),
        "ganancia": float(-objetivos[0]),
        "tiempo_total": float(objetivos[1]),
        "lamina": float(recursos[0]),
        "corte": float(recursos[1]),
        "soldadura": float(recursos[2]),
        "empaque": float(recursos[3]),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(10)

    solucion_factible = np.array([30, 0, 0])
    solucion_no_factible = np.array([60, 50, 40])

    print("Prueba de factibilidad")
    print(
        solucion_factible,
        "Factible:",
        es_factible(solucion_factible),
    )
    print(
        solucion_no_factible,
        "Factible:",
        es_factible(solucion_no_factible),
    )

    print("\nEvaluación de la solución [30, 0, 0]")
    print(resumen_solucion(solucion_factible))

    print("\nPoblación factible de prueba")
    poblacion = generar_poblacion_factible(5, rng)
    print(poblacion)

    print(
        "\nTodas las soluciones son factibles:",
        all(es_factible(x) for x in poblacion),
    )