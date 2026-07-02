"""Implementación sencilla de NSGA-II para el problema de escritorios."""

from __future__ import annotations

import numpy as np

from analisis import (
    clasificacion_no_dominada,
    distancia_crowding,
    evaluar_poblacion,
    extraer_frente_no_dominado,
)
from problema import (
    LIMITES_INFERIORES,
    LIMITES_SUPERIORES,
    es_factible,
    generar_poblacion_factible,
    generar_solucion_factible,
)


def calcular_rangos_y_crowding(
    objetivos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el rango de Pareto y la distancia de crowding
    para cada individuo de una población.
    """
    objetivos = np.asarray(objetivos, dtype=float)

    frentes = clasificacion_no_dominada(objetivos)

    rangos = np.empty(len(objetivos), dtype=int)
    crowding = np.zeros(len(objetivos), dtype=float)

    for rango, frente in enumerate(frentes):
        rangos[frente] = rango

        distancias = distancia_crowding(
            objetivos,
            frente,
        )

        crowding[frente] = distancias

    return rangos, crowding


def torneo_binario(
    poblacion: np.ndarray,
    rangos: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selecciona un individuo mediante torneo binario.

    Se prefiere el individuo con menor rango de Pareto.
    Si ambos tienen el mismo rango, se escoge el que tenga
    mayor distancia de crowding.
    """
    indice_a, indice_b = rng.integers(
        0,
        len(poblacion),
        size=2,
    )

    if rangos[indice_a] < rangos[indice_b]:
        ganador = indice_a

    elif rangos[indice_b] < rangos[indice_a]:
        ganador = indice_b

    elif crowding[indice_a] > crowding[indice_b]:
        ganador = indice_a

    elif crowding[indice_b] > crowding[indice_a]:
        ganador = indice_b

    else:
        ganador = rng.choice(
            [indice_a, indice_b]
        )

    return poblacion[ganador].copy()


def cruce_uniforme(
    padre_1: np.ndarray,
    padre_2: np.ndarray,
    probabilidad_cruce: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica cruce uniforme entre dos padres enteros.

    Cada gen del hijo se toma de uno de los dos padres.
    """
    if rng.random() > probabilidad_cruce:
        return padre_1.copy(), padre_2.copy()

    mascara = rng.random(3) < 0.5

    hijo_1 = np.where(
        mascara,
        padre_1,
        padre_2,
    )

    hijo_2 = np.where(
        mascara,
        padre_2,
        padre_1,
    )

    return (
        hijo_1.astype(int),
        hijo_2.astype(int),
    )


def mutar(
    individuo: np.ndarray,
    probabilidad_mutacion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Aplica mutación por reemplazo aleatorio.

    Cada variable puede recibir un nuevo valor entero dentro
    de su intervalo permitido.
    """
    mutado = individuo.copy()

    for variable in range(len(mutado)):
        if rng.random() < probabilidad_mutacion:
            mutado[variable] = rng.integers(
                LIMITES_INFERIORES[variable],
                LIMITES_SUPERIORES[variable] + 1,
            )

    return mutado.astype(int)


def crear_descendencia(
    poblacion: np.ndarray,
    objetivos: np.ndarray,
    probabilidad_cruce: float,
    probabilidad_mutacion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera una descendencia del mismo tamaño que la población.

    Los hijos no factibles se descartan y se sustituyen por
    nuevas soluciones factibles.
    """
    rangos, crowding = calcular_rangos_y_crowding(
        objetivos
    )

    descendencia = []

    while len(descendencia) < len(poblacion):
        padre_1 = torneo_binario(
            poblacion,
            rangos,
            crowding,
            rng,
        )

        padre_2 = torneo_binario(
            poblacion,
            rangos,
            crowding,
            rng,
        )

        hijo_1, hijo_2 = cruce_uniforme(
            padre_1,
            padre_2,
            probabilidad_cruce,
            rng,
        )

        hijo_1 = mutar(
            hijo_1,
            probabilidad_mutacion,
            rng,
        )

        hijo_2 = mutar(
            hijo_2,
            probabilidad_mutacion,
            rng,
        )

        for hijo in (hijo_1, hijo_2):
            if len(descendencia) >= len(poblacion):
                break

            if es_factible(hijo):
                descendencia.append(hijo)

            else:
                descendencia.append(
                    generar_solucion_factible(rng)
                )

    return np.array(descendencia, dtype=int)


def seleccion_ambiental(
    poblacion_combinada: np.ndarray,
    objetivos_combinados: np.ndarray,
    tamano_poblacion: int,
) -> np.ndarray:
    """
    Selecciona la siguiente generación mediante rangos de Pareto
    y distancia de crowding.
    """
    frentes = clasificacion_no_dominada(
        objetivos_combinados
    )

    indices_seleccionados = []

    for frente in frentes:
        espacio_disponible = (
            tamano_poblacion
            - len(indices_seleccionados)
        )

        if espacio_disponible <= 0:
            break

        if len(frente) <= espacio_disponible:
            indices_seleccionados.extend(
                frente.tolist()
            )

        else:
            distancias = distancia_crowding(
                objetivos_combinados,
                frente,
            )

            orden = np.argsort(
                -distancias,
                kind="mergesort",
            )

            seleccion_parcial = frente[
                orden[:espacio_disponible]
            ]

            indices_seleccionados.extend(
                seleccion_parcial.tolist()
            )

            break

    indices_seleccionados = np.array(
        indices_seleccionados,
        dtype=int,
    )

    return poblacion_combinada[
        indices_seleccionados
    ].copy()


def ejecutar_nsga2(
    tamano_poblacion: int = 30,
    iteraciones: int = 100,
    probabilidad_cruce: float = 0.90,
    probabilidad_mutacion: float = 1.0 / 3.0,
    semilla: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Ejecuta NSGA-II y devuelve:

    1. Población final.
    2. Soluciones del frente no dominado.
    3. Objetivos del frente no dominado.
    """
    if tamano_poblacion <= 0:
        raise ValueError(
            "El tamaño de la población debe ser positivo."
        )

    if iteraciones <= 0:
        raise ValueError(
            "El número de iteraciones debe ser positivo."
        )

    if not 0.0 <= probabilidad_cruce <= 1.0:
        raise ValueError(
            "La probabilidad de cruce debe estar entre 0 y 1."
        )

    if not 0.0 <= probabilidad_mutacion <= 1.0:
        raise ValueError(
            "La probabilidad de mutación debe estar entre 0 y 1."
        )

    rng = np.random.default_rng(semilla)

    poblacion = generar_poblacion_factible(
        tamano_poblacion,
        rng,
    )

    for _ in range(iteraciones):
        objetivos = evaluar_poblacion(poblacion)

        descendencia = crear_descendencia(
            poblacion,
            objetivos,
            probabilidad_cruce,
            probabilidad_mutacion,
            rng,
        )

        poblacion_combinada = np.vstack(
            (poblacion, descendencia)
        )

        objetivos_combinados = evaluar_poblacion(
            poblacion_combinada
        )

        poblacion = seleccion_ambiental(
            poblacion_combinada,
            objetivos_combinados,
            tamano_poblacion,
        )

    soluciones_frente, objetivos_frente = (
        extraer_frente_no_dominado(
            poblacion,
            eliminar_duplicados=True,
        )
    )

    return (
        poblacion,
        soluciones_frente,
        objetivos_frente,
    )


if __name__ == "__main__":
    np.set_printoptions(
        suppress=True,
        precision=2,
    )

    TAMANO_POBLACION = 30
    ITERACIONES = 100
    PROBABILIDAD_CRUCE = 0.90
    PROBABILIDAD_MUTACION = 1.0 / 3.0
    SEMILLA = 10

    poblacion_final, soluciones_frente, objetivos_frente = (
        ejecutar_nsga2(
            tamano_poblacion=TAMANO_POBLACION,
            iteraciones=ITERACIONES,
            probabilidad_cruce=PROBABILIDAD_CRUCE,
            probabilidad_mutacion=PROBABILIDAD_MUTACION,
            semilla=SEMILLA,
        )
    )

    todas_factibles = all(
        es_factible(individuo)
        for individuo in poblacion_final
    )

    resultados_frente = np.column_stack(
        (
            soluciones_frente,
            -objetivos_frente[:, 0],
            objetivos_frente[:, 1],
        )
    )

    print("Ejecución de NSGA-II")
    print("Tamaño de la población:", len(poblacion_final))
    print("Iteraciones:", ITERACIONES)
    print(
        "Probabilidad de cruce:",
        PROBABILIDAD_CRUCE,
    )
    print(
        "Probabilidad de mutación:",
        round(PROBABILIDAD_MUTACION, 4),
    )
    print(
        "Todas las soluciones son factibles:",
        todas_factibles,
    )
    print(
        "Soluciones no dominadas únicas:",
        len(soluciones_frente),
    )

    print(
        "\nFrente obtenido: "
        "[x1, x2, x3, ganancia, tiempo]"
    )
    print(resultados_frente)