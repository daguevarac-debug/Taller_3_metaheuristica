"""Implementación sencilla de MOPSO para el problema de escritorios."""

from __future__ import annotations

import numpy as np

from analisis import (
    distancia_crowding,
    domina,
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


def actualizar_repositorio(
    repositorio: np.ndarray,
    nuevas_soluciones: np.ndarray,
    tamano_maximo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Actualiza el repositorio externo con soluciones no dominadas.

    Si el repositorio supera el tamaño máximo, se conservan las
    soluciones con mayor distancia de crowding.
    """
    nuevas_soluciones = np.asarray(
        nuevas_soluciones,
        dtype=int,
    )

    if repositorio.size == 0:
        soluciones_combinadas = nuevas_soluciones.copy()
    else:
        soluciones_combinadas = np.vstack(
            (
                repositorio,
                nuevas_soluciones,
            )
        )

    soluciones_frente, objetivos_frente = (
        extraer_frente_no_dominado(
            soluciones_combinadas,
            eliminar_duplicados=True,
        )
    )

    if len(soluciones_frente) > tamano_maximo:
        indices = np.arange(len(soluciones_frente))

        distancias = distancia_crowding(
            objetivos_frente,
            indices,
        )

        orden = np.argsort(
            -distancias,
            kind="mergesort",
        )

        seleccionados = orden[:tamano_maximo]

        soluciones_frente = soluciones_frente[
            seleccionados
        ]

        objetivos_frente = objetivos_frente[
            seleccionados
        ]

    orden_tiempo = np.argsort(
        objetivos_frente[:, 1],
        kind="mergesort",
    )

    return (
        soluciones_frente[orden_tiempo],
        objetivos_frente[orden_tiempo],
    )


def seleccionar_lider(
    repositorio: np.ndarray,
    objetivos_repositorio: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Selecciona un líder mediante torneo basado en crowding.

    Se favorecen las soluciones ubicadas en regiones menos
    congestionadas del frente de Pareto.
    """
    cantidad = len(repositorio)

    if cantidad == 0:
        raise ValueError(
            "El repositorio no puede estar vacío."
        )

    if cantidad == 1:
        return repositorio[0].copy()

    indices = np.arange(cantidad)

    distancias = distancia_crowding(
        objetivos_repositorio,
        indices,
    )

    indice_a, indice_b = rng.choice(
        cantidad,
        size=2,
        replace=False,
    )

    if distancias[indice_a] > distancias[indice_b]:
        ganador = indice_a

    elif distancias[indice_b] > distancias[indice_a]:
        ganador = indice_b

    else:
        ganador = rng.choice(
            [indice_a, indice_b]
        )

    return repositorio[ganador].copy()


def actualizar_mejores_personales(
    posiciones: np.ndarray,
    objetivos_actuales: np.ndarray,
    mejores_personales: np.ndarray,
    objetivos_mejores: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Actualiza la mejor posición personal de cada partícula.

    Si ninguna solución domina a la otra, se elige una de ellas
    aleatoriamente para mantener diversidad.
    """
    nuevos_mejores = mejores_personales.copy()
    nuevos_objetivos = objetivos_mejores.copy()

    for i in range(len(posiciones)):
        objetivo_actual = objetivos_actuales[i]
        objetivo_mejor = objetivos_mejores[i]

        if domina(objetivo_actual, objetivo_mejor):
            nuevos_mejores[i] = posiciones[i]
            nuevos_objetivos[i] = objetivo_actual

        elif domina(objetivo_mejor, objetivo_actual):
            continue

        elif not np.allclose(
            objetivo_actual,
            objetivo_mejor,
        ):
            if rng.random() < 0.5:
                nuevos_mejores[i] = posiciones[i]
                nuevos_objetivos[i] = objetivo_actual

    return nuevos_mejores, nuevos_objetivos


def ejecutar_mopso(
    tamano_enjambre: int = 30,
    iteraciones: int = 100,
    inercia_inicial: float = 0.90,
    inercia_final: float = 0.40,
    coeficiente_cognitivo: float = 1.50,
    coeficiente_social: float = 1.50,
    tamano_repositorio: int = 100,
    semilla: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Ejecuta MOPSO y devuelve:

    1. Enjambre final.
    2. Soluciones del repositorio no dominado.
    3. Objetivos del repositorio no dominado.
    """
    if tamano_enjambre <= 0:
        raise ValueError(
            "El tamaño del enjambre debe ser positivo."
        )

    if iteraciones <= 0:
        raise ValueError(
            "El número de iteraciones debe ser positivo."
        )

    if tamano_repositorio <= 0:
        raise ValueError(
            "El tamaño del repositorio debe ser positivo."
        )

    if inercia_inicial < inercia_final:
        raise ValueError(
            "La inercia inicial debe ser mayor o igual "
            "que la inercia final."
        )

    rng = np.random.default_rng(semilla)

    posiciones = generar_poblacion_factible(
        tamano_enjambre,
        rng,
    ).astype(float)

    velocidades = np.zeros_like(
        posiciones,
        dtype=float,
    )

    rango_variables = (
        LIMITES_SUPERIORES
        - LIMITES_INFERIORES
    ).astype(float)

    velocidad_maxima = 0.20 * rango_variables
    velocidad_minima = -velocidad_maxima

    objetivos_actuales = evaluar_poblacion(
        posiciones.astype(int)
    )

    mejores_personales = posiciones.copy()
    objetivos_mejores = objetivos_actuales.copy()

    repositorio = np.empty(
        (0, 3),
        dtype=int,
    )

    repositorio, objetivos_repositorio = (
        actualizar_repositorio(
            repositorio,
            posiciones.astype(int),
            tamano_repositorio,
        )
    )

    for iteracion in range(iteraciones):
        if iteraciones == 1:
            inercia = inercia_final
        else:
            fraccion = iteracion / (iteraciones - 1)

            inercia = (
                inercia_inicial
                - (
                    inercia_inicial
                    - inercia_final
                )
                * fraccion
            )

        for i in range(tamano_enjambre):
            lider = seleccionar_lider(
                repositorio,
                objetivos_repositorio,
                rng,
            ).astype(float)

            aleatorio_cognitivo = rng.random(3)
            aleatorio_social = rng.random(3)

            componente_inercia = (
                inercia * velocidades[i]
            )

            componente_cognitivo = (
                coeficiente_cognitivo
                * aleatorio_cognitivo
                * (
                    mejores_personales[i]
                    - posiciones[i]
                )
            )

            componente_social = (
                coeficiente_social
                * aleatorio_social
                * (
                    lider
                    - posiciones[i]
                )
            )

            velocidades[i] = (
                componente_inercia
                + componente_cognitivo
                + componente_social
            )

            velocidades[i] = np.clip(
                velocidades[i],
                velocidad_minima,
                velocidad_maxima,
            )

            nueva_posicion = (
                posiciones[i]
                + velocidades[i]
            )

            nueva_posicion = np.clip(
                nueva_posicion,
                LIMITES_INFERIORES,
                LIMITES_SUPERIORES,
            )

            nueva_posicion = np.rint(
                nueva_posicion
            ).astype(int)

            if es_factible(nueva_posicion):
                posiciones[i] = nueva_posicion

            else:
                posiciones[i] = (
                    generar_solucion_factible(rng)
                )

                velocidades[i] = 0.0

        posiciones_enteras = posiciones.astype(int)

        objetivos_actuales = evaluar_poblacion(
            posiciones_enteras
        )

        (
            mejores_personales,
            objetivos_mejores,
        ) = actualizar_mejores_personales(
            posiciones,
            objetivos_actuales,
            mejores_personales,
            objetivos_mejores,
            rng,
        )

        (
            repositorio,
            objetivos_repositorio,
        ) = actualizar_repositorio(
            repositorio,
            posiciones_enteras,
            tamano_repositorio,
        )

    enjambre_final = posiciones.astype(int)

    return (
        enjambre_final,
        repositorio,
        objetivos_repositorio,
    )


if __name__ == "__main__":
    np.set_printoptions(
        suppress=True,
        precision=2,
    )

    TAMANO_ENJAMBRE = 30
    ITERACIONES = 100
    INERCIA_INICIAL = 0.90
    INERCIA_FINAL = 0.40
    COEFICIENTE_COGNITIVO = 1.50
    COEFICIENTE_SOCIAL = 1.50
    TAMANO_REPOSITORIO = 100
    SEMILLA = 10

    (
        enjambre_final,
        soluciones_repositorio,
        objetivos_repositorio,
    ) = ejecutar_mopso(
        tamano_enjambre=TAMANO_ENJAMBRE,
        iteraciones=ITERACIONES,
        inercia_inicial=INERCIA_INICIAL,
        inercia_final=INERCIA_FINAL,
        coeficiente_cognitivo=COEFICIENTE_COGNITIVO,
        coeficiente_social=COEFICIENTE_SOCIAL,
        tamano_repositorio=TAMANO_REPOSITORIO,
        semilla=SEMILLA,
    )

    todas_factibles = all(
        es_factible(particula)
        for particula in enjambre_final
    )

    resultados_repositorio = np.column_stack(
        (
            soluciones_repositorio,
            -objetivos_repositorio[:, 0],
            objetivos_repositorio[:, 1],
        )
    )

    print("Ejecución de MOPSO")
    print(
        "Tamaño del enjambre:",
        len(enjambre_final),
    )
    print("Iteraciones:", ITERACIONES)
    print(
        "Inercia inicial:",
        INERCIA_INICIAL,
    )
    print(
        "Inercia final:",
        INERCIA_FINAL,
    )
    print(
        "Coeficiente cognitivo:",
        COEFICIENTE_COGNITIVO,
    )
    print(
        "Coeficiente social:",
        COEFICIENTE_SOCIAL,
    )
    print(
        "Tamaño máximo del repositorio:",
        TAMANO_REPOSITORIO,
    )
    print(
        "Todas las soluciones son factibles:",
        todas_factibles,
    )
    print(
        "Soluciones no dominadas únicas:",
        len(soluciones_repositorio),
    )

    print(
        "\nRepositorio final: "
        "[x1, x2, x3, ganancia, tiempo]"
    )
    print(resultados_repositorio)