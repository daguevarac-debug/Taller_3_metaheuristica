"""Funciones comunes para el análisis multiobjetivo."""

from __future__ import annotations

import numpy as np

from problema import es_factible, evaluar, generar_poblacion_factible


def domina(
    objetivos_a: np.ndarray,
    objetivos_b: np.ndarray,
) -> bool:
    """
    Determina si la solución A domina a la solución B.

    Ambos objetivos se minimizan. A domina a B cuando:
    1. A no es peor que B en ningún objetivo.
    2. A es estrictamente mejor que B en al menos un objetivo.
    """
    objetivos_a = np.asarray(objetivos_a, dtype=float)
    objetivos_b = np.asarray(objetivos_b, dtype=float)

    if objetivos_a.shape != objetivos_b.shape:
        raise ValueError(
            "Los vectores de objetivos deben tener la misma dimensión."
        )

    no_es_peor = np.all(objetivos_a <= objetivos_b)
    es_mejor_en_alguno = np.any(objetivos_a < objetivos_b)

    return bool(no_es_peor and es_mejor_en_alguno)


def evaluar_poblacion(
    poblacion: np.ndarray,
) -> np.ndarray:
    """Evalúa todos los individuos de una población factible."""
    poblacion = np.asarray(poblacion)

    if poblacion.ndim != 2 or poblacion.shape[1] != 3:
        raise ValueError(
            "La población debe tener dimensiones (n_individuos, 3)."
        )

    if not all(
        es_factible(individuo)
        for individuo in poblacion
    ):
        raise ValueError(
            "La población contiene al menos una solución no factible."
        )

    return np.array(
        [
            evaluar(individuo)
            for individuo in poblacion
        ],
        dtype=float,
    )


def mascara_no_dominados(
    objetivos: np.ndarray,
) -> np.ndarray:
    """
    Identifica las soluciones que pertenecen al frente no dominado.

    True representa una solución no dominada.
    """
    objetivos = np.asarray(objetivos, dtype=float)

    if objetivos.ndim != 2 or objetivos.shape[1] != 2:
        raise ValueError(
            "La matriz de objetivos debe tener dimensiones "
            "(n_soluciones, 2)."
        )

    numero_soluciones = len(objetivos)
    mascara = np.ones(numero_soluciones, dtype=bool)

    for i in range(numero_soluciones):
        for j in range(numero_soluciones):
            if (
                i != j
                and domina(objetivos[j], objetivos[i])
            ):
                mascara[i] = False
                break

    return mascara


def extraer_frente_no_dominado(
    poblacion: np.ndarray,
    eliminar_duplicados: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae las soluciones no dominadas y sus objetivos.

    Las soluciones se ordenan de menor a mayor tiempo.
    """
    poblacion = np.asarray(poblacion, dtype=int)
    objetivos = evaluar_poblacion(poblacion)

    mascara = mascara_no_dominados(objetivos)

    soluciones_frente = poblacion[mascara]
    objetivos_frente = objetivos[mascara]

    if eliminar_duplicados and len(soluciones_frente) > 0:
        datos_completos = np.column_stack(
            (
                soluciones_frente,
                objetivos_frente,
            )
        )

        _, indices_unicos = np.unique(
            datos_completos,
            axis=0,
            return_index=True,
        )

        indices_unicos = np.sort(indices_unicos)

        soluciones_frente = soluciones_frente[
            indices_unicos
        ]

        objetivos_frente = objetivos_frente[
            indices_unicos
        ]

    if len(objetivos_frente) == 0:
        return soluciones_frente, objetivos_frente

    orden = np.argsort(
        objetivos_frente[:, 1],
        kind="mergesort",
    )

    return (
        soluciones_frente[orden],
        objetivos_frente[orden],
    )


def clasificacion_no_dominada(
    objetivos: np.ndarray,
) -> list[np.ndarray]:
    """
    Clasifica las soluciones en frentes sucesivos de Pareto.

    El primer frente contiene las soluciones no dominadas.
    """
    objetivos = np.asarray(objetivos, dtype=float)

    if objetivos.ndim != 2 or objetivos.shape[1] != 2:
        raise ValueError(
            "La matriz de objetivos debe tener dimensiones "
            "(n_soluciones, 2)."
        )

    numero_soluciones = len(objetivos)

    soluciones_dominadas = [
        []
        for _ in range(numero_soluciones)
    ]

    numero_dominadores = np.zeros(
        numero_soluciones,
        dtype=int,
    )

    for p in range(numero_soluciones):
        for q in range(numero_soluciones):
            if p == q:
                continue

            if domina(
                objetivos[p],
                objetivos[q],
            ):
                soluciones_dominadas[p].append(q)

            elif domina(
                objetivos[q],
                objetivos[p],
            ):
                numero_dominadores[p] += 1

    frente_actual = [
        indice
        for indice in range(numero_soluciones)
        if numero_dominadores[indice] == 0
    ]

    frentes = []

    while frente_actual:
        frentes.append(
            np.array(
                frente_actual,
                dtype=int,
            )
        )

        siguiente_frente = []

        for p in frente_actual:
            for q in soluciones_dominadas[p]:
                numero_dominadores[q] -= 1

                if numero_dominadores[q] == 0:
                    siguiente_frente.append(q)

        frente_actual = siguiente_frente

    return frentes


def distancia_crowding(
    objetivos: np.ndarray,
    indices_frente: np.ndarray,
) -> np.ndarray:
    """
    Calcula la distancia de crowding de un frente.

    Las soluciones extremas reciben distancia infinita.
    """
    objetivos = np.asarray(objetivos, dtype=float)

    indices_frente = np.asarray(
        indices_frente,
        dtype=int,
    )

    if objetivos.ndim != 2 or objetivos.shape[1] != 2:
        raise ValueError(
            "La matriz de objetivos debe tener dimensiones "
            "(n_soluciones, 2)."
        )

    if indices_frente.ndim != 1:
        raise ValueError(
            "Los índices del frente deben formar un vector."
        )

    numero_elementos = len(indices_frente)

    if numero_elementos == 0:
        return np.array([], dtype=float)

    distancias = np.zeros(
        numero_elementos,
        dtype=float,
    )

    if numero_elementos <= 2:
        distancias[:] = np.inf
        return distancias

    objetivos_frente = objetivos[
        indices_frente
    ]

    numero_objetivos = objetivos.shape[1]

    for objetivo in range(numero_objetivos):
        orden = np.argsort(
            objetivos_frente[:, objetivo],
            kind="mergesort",
        )

        distancias[orden[0]] = np.inf
        distancias[orden[-1]] = np.inf

        valor_minimo = objetivos_frente[
            orden[0],
            objetivo,
        ]

        valor_maximo = objetivos_frente[
            orden[-1],
            objetivo,
        ]

        rango = valor_maximo - valor_minimo

        if np.isclose(rango, 0.0):
            continue

        for posicion in range(
            1,
            numero_elementos - 1,
        ):
            indice_anterior = orden[
                posicion - 1
            ]

            indice_siguiente = orden[
                posicion + 1
            ]

            indice_actual = orden[posicion]

            diferencia_normalizada = (
                objetivos_frente[
                    indice_siguiente,
                    objetivo,
                ]
                - objetivos_frente[
                    indice_anterior,
                    objetivo,
                ]
            ) / rango

            distancias[indice_actual] += (
                diferencia_normalizada
            )

    return distancias


def normalizar_objetivos(
    objetivos: np.ndarray,
) -> np.ndarray:
    """Normaliza cada objetivo entre 0 y 1 mediante min-max."""
    objetivos = np.asarray(objetivos, dtype=float)

    if objetivos.ndim != 2:
        raise ValueError(
            "Los objetivos deben estar en una matriz."
        )

    if len(objetivos) == 0:
        raise ValueError(
            "La matriz de objetivos no puede estar vacía."
        )

    minimos = np.min(
        objetivos,
        axis=0,
    )

    maximos = np.max(
        objetivos,
        axis=0,
    )

    rangos = maximos - minimos

    rangos_seguros = np.where(
        np.isclose(rangos, 0.0),
        1.0,
        rangos,
    )

    return (
        objetivos - minimos
    ) / rangos_seguros


def porcentaje_no_dominadas(
    objetivos: np.ndarray,
) -> float:
    """
    Calcula el porcentaje de soluciones no dominadas.

    El resultado se expresa entre 0 y 100.
    """
    objetivos = np.asarray(objetivos, dtype=float)

    if objetivos.ndim != 2 or objetivos.shape[1] != 2:
        raise ValueError(
            "La matriz de objetivos debe tener dimensiones "
            "(n_soluciones, 2)."
        )

    if len(objetivos) == 0:
        raise ValueError(
            "La matriz de objetivos no puede estar vacía."
        )

    mascara = mascara_no_dominados(objetivos)

    return float(
        100.0
        * np.sum(mascara)
        / len(objetivos)
    )


def calcular_spacing(
    objetivos_frente: np.ndarray,
) -> float:
    """
    Calcula el spacing de un frente de Pareto.

    Un valor bajo indica una distribución más uniforme.
    """
    objetivos_frente = np.asarray(
        objetivos_frente,
        dtype=float,
    )

    if (
        objetivos_frente.ndim != 2
        or objetivos_frente.shape[1] != 2
    ):
        raise ValueError(
            "El frente debe tener dimensiones "
            "(n_soluciones, 2)."
        )

    # Se eliminan objetivos repetidos antes de medir
    # la distribución del frente.
    objetivos_unicos = np.unique(
        objetivos_frente,
        axis=0,
    )

    numero_soluciones = len(
        objetivos_unicos
    )

    if numero_soluciones <= 1:
        return 0.0

    objetivos_normalizados = (
        normalizar_objetivos(
            objetivos_unicos
        )
    )

    distancias_minimas = np.zeros(
        numero_soluciones,
        dtype=float,
    )

    for i in range(numero_soluciones):
        distancias = np.sum(
            np.abs(
                objetivos_normalizados
                - objetivos_normalizados[i]
            ),
            axis=1,
        )

        distancias[i] = np.inf

        distancias_minimas[i] = np.min(
            distancias
        )

    distancia_media = np.mean(
        distancias_minimas
    )

    spacing = np.sqrt(
        np.sum(
            (
                distancias_minimas
                - distancia_media
            )
            ** 2
        )
        / (numero_soluciones - 1)
    )

    return float(spacing)


if __name__ == "__main__":
    np.set_printoptions(
        suppress=True,
        precision=4,
    )

    # Prueba controlada de dominancia
    objetivos_prueba = np.array(
        [
            [-1200.0, 24.0],
            [-1300.0, 24.0],
            [-1400.0, 30.0],
            [-1100.0, 26.0],
        ]
    )

    print("Prueba de dominancia")

    print(
        "La solución 2 domina a la solución 1:",
        domina(
            objetivos_prueba[1],
            objetivos_prueba[0],
        ),
    )

    print(
        "La solución 1 domina a la solución 2:",
        domina(
            objetivos_prueba[0],
            objetivos_prueba[1],
        ),
    )

    mascara = mascara_no_dominados(
        objetivos_prueba
    )

    print(
        "\nMáscara de soluciones no dominadas"
    )
    print(mascara)

    print("\nObjetivos no dominados")
    print(objetivos_prueba[mascara])

    # Prueba con población factible
    rng = np.random.default_rng(10)

    poblacion = generar_poblacion_factible(
        30,
        rng,
    )

    (
        soluciones_frente,
        objetivos_frente,
    ) = extraer_frente_no_dominado(
        poblacion
    )

    print("\nPrueba con población factible")
    print(
        "Tamaño de la población:",
        len(poblacion),
    )

    print(
        "Número de soluciones no dominadas:",
        len(soluciones_frente),
    )

    print("\nSoluciones del frente")
    print(soluciones_frente)

    print(
        "\nObjetivos del frente: "
        "[f1 = -G, f2 = T]"
    )
    print(objetivos_frente)

    # Prueba de clasificación por frentes
    print(
        "\nClasificación completa por frentes"
    )

    frentes_prueba = (
        clasificacion_no_dominada(
            objetivos_prueba
        )
    )

    for numero, frente in enumerate(
        frentes_prueba,
        start=1,
    ):
        print(
            f"Frente {numero}: {frente}"
        )

    # Prueba de distancia de crowding
    objetivos_crowding = np.array(
        [
            [-1000.0, 20.0],
            [-1200.0, 25.0],
            [-1400.0, 30.0],
            [-1600.0, 35.0],
            [-1800.0, 40.0],
        ]
    )

    indices_crowding = np.arange(
        len(objetivos_crowding)
    )

    distancias = distancia_crowding(
        objetivos_crowding,
        indices_crowding,
    )

    print(
        "\nPrueba de distancia de crowding"
    )
    print(distancias)

    # Prueba de normalización
    objetivos_normalizados = (
        normalizar_objetivos(
            objetivos_crowding
        )
    )

    print("\nPrueba de normalización")
    print(objetivos_normalizados)

    # Prueba de indicadores
    objetivos_indicadores = np.array(
        [
            [-1000.0, 20.0],
            [-1200.0, 25.0],
            [-1400.0, 30.0],
            [-900.0, 30.0],
        ]
    )

    porcentaje = porcentaje_no_dominadas(
        objetivos_indicadores
    )

    mascara_indicadores = (
        mascara_no_dominados(
            objetivos_indicadores
        )
    )

    frente_indicadores = (
        objetivos_indicadores[
            mascara_indicadores
        ]
    )

    spacing = calcular_spacing(
        frente_indicadores
    )

    print("\nPrueba de indicadores")

    print(
        "Porcentaje de soluciones no dominadas:",
        porcentaje,
    )

    print(
        "Spacing:",
        spacing,
    )