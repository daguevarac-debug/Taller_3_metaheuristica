# Taller 3 – Optimización metaheurística multiobjetivo

## 6. Parámetros mínimos de simulación

Los algoritmos NSGA-II y MOPSO se ejecutaron bajo las mismas condiciones generales. En ambos casos se utilizó una población o enjambre de 30 soluciones y un total de 100 iteraciones. Además, se realizaron cinco corridas independientes mediante las semillas aleatorias 10, 20, 30, 40 y 50, lo que permitió evaluar el comportamiento de los métodos bajo diferentes poblaciones iniciales.

Para NSGA-II se utilizó una probabilidad de cruce de 0,90 y una probabilidad de mutación de 1/3. La selección de los padres se realizó mediante torneo binario, dando prioridad al individuo con menor rango de Pareto y, en caso de igualdad, al que presentara mayor distancia de crowding. La selección de supervivientes se efectuó mediante clasificación no dominada y distancia de crowding.

Para MOPSO se empleó un factor de inercia decreciente desde 0,90 hasta 0,40. Los coeficientes cognitivo y social se establecieron en 1,50. El repositorio externo tuvo una capacidad máxima de 100 soluciones no dominadas. El líder social de cada partícula se seleccionó mediante un torneo basado en la distancia de crowding, favoreciendo las soluciones localizadas en regiones menos congestionadas del frente.

En todas las corridas se verificó que las soluciones finales fueran enteras, no negativas y factibles respecto a las restricciones de demanda, producción mínima y disponibilidad de recursos.

## 7. Datos que deben asumir o definir

### 7.1. Cómo se codifican las variables enteras

Las variables de decisión se codifican mediante un vector de tres componentes:

$$
\mathbf{x}=[x_1,x_2,x_3]
$$

donde \(x_1\), \(x_2\) y \(x_3\) representan, respectivamente, las cantidades de escritorios básicos, estándar y premium. Cada componente toma valores enteros no negativos y respeta los límites \(0\leq x_1\leq60\), \(0\leq x_2\leq50\) y \(0\leq x_3\leq40\).

En NSGA-II, los individuos se generan y almacenan directamente como vectores enteros. Los operadores de cruce y mutación conservan esta representación. En MOPSO, las partículas actualizan temporalmente sus posiciones mediante valores reales, pero antes de evaluar una solución cada componente se limita al intervalo permitido y se redondea al entero más cercano.

Esta implementación se encuentra principalmente en `problema.py`, mediante las constantes `LIMITES_INFERIORES`, `LIMITES_SUPERIORES` y la función `es_vector_entero()`. La conservación de valores enteros en NSGA-II se realiza en `nsga2.py`, especialmente en las funciones `cruce_uniforme()` y `mutar()`. En MOPSO se implementa en `mopso.py`, donde la nueva posición se redondea mediante `np.rint(...).astype(int)`.

### 7.2. Cómo se manejan las soluciones no factibles

Las soluciones que incumplen alguna restricción se descartan, tal como exige el enunciado. No se emplean funciones de penalización.

La factibilidad se verifica en `problema.py` mediante la función `es_factible()`, la cual comprueba los límites de las variables, la producción mínima y la disponibilidad de lámina, corte, soldadura y empaque.

En NSGA-II, cuando un hijo no es factible, se reemplaza por una nueva solución factible generada aleatoriamente. En MOPSO se aplica el mismo criterio; además, la velocidad de la partícula se reinicia en cero. De esta manera, ambos algoritmos trabajan únicamente con soluciones válidas.

### 7.3. Cómo se identifican las soluciones no dominadas

Una solución se considera no dominada cuando ninguna otra solución es igual o mejor en todos los objetivos y estrictamente mejor en al menos uno. Como ambos objetivos se expresan en forma de minimización, una solución A domina a una solución B cuando sus valores objetivo son menores o iguales y al menos uno es estrictamente menor.

Este procedimiento se implementa en `analisis.py` mediante las funciones `domina()`, `mascara_no_dominados()` y `clasificacion_no_dominada()`. La primera compara dos soluciones, mientras que las demás identifican el frente de Pareto y clasifican la población en frentes sucesivos.

### 7.4. Cómo se normalizan los objetivos

Los objetivos se normalizan mediante el método min-max:

$$
z_{ij}=\frac{f_{ij}-f_j^{\min}}{f_j^{\max}-f_j^{\min}}
$$

Con esta transformación, cada objetivo queda expresado entre 0 y 1. La normalización evita que la ganancia y el tiempo tengan una influencia desigual debido a sus diferentes escalas numéricas.

La implementación se encuentra en `analisis.py`, dentro de la función `normalizar_objetivos()`. Esta función se empleará para calcular el indicador spacing y para aplicar las técnicas de decisión multicriterio.

### 7.5. Parámetros utilizados y procedencia de los códigos

NSGA-II se ejecutó con una población de 30 individuos, 100 iteraciones, una probabilidad de cruce de 0,90 y una probabilidad de mutación de 1/3. La selección de padres se realizó mediante torneo binario, mientras que la selección ambiental empleó clasificación no dominada y distancia de crowding.

MOPSO se ejecutó con 30 partículas y 100 iteraciones. Se utilizó un factor de inercia decreciente de 0,90 a 0,40, coeficientes cognitivo y social de 1,50 y un repositorio externo con capacidad máxima de 100 soluciones. El líder se seleccionó favoreciendo las regiones con mayor distancia de crowding.

Los códigos fueron desarrollados por el grupo en Python con apoyo de inteligencia artificial generativa, específicamente ChatGPT, para estructurar, revisar y explicar las funciones. No se copiaron implementaciones completas de repositorios externos. Los conceptos generales de algoritmo genético y PSO se tomaron del material suministrado en el curso. El grupo ejecutó y verificó cada componente mediante pruebas de factibilidad, dominancia y reproducibilidad.

### 7.6. Pesos utilizados en las técnicas de decisión multicriterio

Se asignó el mismo peso a los dos objetivos:

$$
w_G=0.50
$$

$$
w_T=0.50
$$

El peso de 0,50 para la ganancia y de 0,50 para el tiempo representa una preferencia equilibrada entre aumentar la rentabilidad y reducir el tiempo de producción. Esta decisión se adoptó porque el enunciado no establece que uno de los objetivos tenga mayor importancia que el otro.

Los mismos pesos se aplicarán en TOPSIS y en la suma ponderada normalizada, lo que permitirá comparar las técnicas bajo condiciones equivalentes.

## 8. Indicadores de calidad del frente de Pareto

El porcentaje de soluciones no dominadas corresponde a la proporción de soluciones finales que pertenecen al frente de Pareto. Un valor alto indica que el algoritmo conserva una mayor cantidad de alternativas eficientes dentro de su población final.

El indicador spacing mide la uniformidad con la que las soluciones se distribuyen sobre el frente. Para calcularlo se normalizan los objetivos, se determina para cada solución la distancia mínima respecto a las demás y se evalúa la dispersión de esas distancias. En consecuencia, un valor bajo indica una distribución más uniforme.

| Método | Porcentaje promedio de soluciones no dominadas (%) | Desviación del porcentaje | Spacing promedio | Desviación del spacing |
|---|---:|---:|---:|---:|
| NSGA-II | 100,000000 | 0,000000 | 0,024998 | 0,002935 |
| MOPSO | 91,333333 | 6,497863 | 0,044829 | 0,013013 |

NSGA-II presentó el mejor desempeño bajo los dos indicadores exigidos. Alcanzó un 100 % promedio de soluciones no dominadas en las cinco corridas, mientras que MOPSO obtuvo un promedio de 91,333333 %. Además, NSGA-II registró un spacing promedio menor, lo que evidencia una distribución más uniforme del frente.

La variabilidad también fue menor en NSGA-II, puesto que la desviación del porcentaje fue igual a cero y la desviación del spacing fue inferior a la obtenida por MOPSO. Aunque MOPSO alcanzó un spacing menor en la corrida con semilla 30, su comportamiento fue menos estable en el conjunto de las cinco ejecuciones. Como resultado complementario, NSGA-II también presentó un menor tiempo promedio de ejecución, con 3,102449 s frente a 5,836887 s de MOPSO. Por tanto, NSGA-II produjo el mejor frente de Pareto según los indicadores analizados.
