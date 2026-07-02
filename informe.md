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

Para el indicador spacing se emplea la función `normalizar_objetivos()` de `analisis.py`. En las técnicas multicriterio se usa una normalización min-max orientada, de modo que un valor alto represente mejor desempeño: la ganancia se trata como criterio de beneficio y el tiempo como criterio de costo.

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

Los mismos pesos se aplicaron en TOPSIS y en la suma ponderada normalizada, lo que permitió comparar las técnicas bajo condiciones equivalentes.

## 8. Indicadores de calidad del frente de Pareto

El porcentaje de soluciones no dominadas corresponde a la proporción de soluciones finales que pertenecen al frente de Pareto. Un valor alto indica que el algoritmo conserva una mayor cantidad de alternativas eficientes dentro de su población final.

El indicador spacing mide la uniformidad con la que las soluciones se distribuyen sobre el frente. Para calcularlo se normalizan los objetivos, se determina para cada solución la distancia mínima respecto a las demás y se evalúa la dispersión de esas distancias. En consecuencia, un valor bajo indica una distribución más uniforme.

| Método | Porcentaje promedio de soluciones no dominadas (%) | Desviación del porcentaje | Spacing promedio | Desviación del spacing |
|---|---:|---:|---:|---:|
| NSGA-II | 100,000000 | 0,000000 | 0,024998 | 0,002935 |
| MOPSO | 91,333333 | 6,497863 | 0,044829 | 0,013013 |

NSGA-II presentó el mejor desempeño bajo los dos indicadores exigidos. Alcanzó un 100 % promedio de soluciones no dominadas en las cinco corridas, mientras que MOPSO obtuvo un promedio de 91,333333 %. Además, NSGA-II registró un spacing promedio menor, lo que evidencia una distribución más uniforme del frente.

La variabilidad también fue menor en NSGA-II, puesto que la desviación del porcentaje fue igual a cero y la desviación del spacing fue inferior a la obtenida por MOPSO. Aunque MOPSO alcanzó un spacing menor en la corrida con semilla 30, su comportamiento fue menos estable en el conjunto de las cinco ejecuciones. Como resultado complementario, NSGA-II también presentó un menor tiempo promedio de ejecución, con 3,102449 s frente a 5,836887 s de MOPSO. Por tanto, NSGA-II produjo el mejor frente de Pareto según los indicadores analizados.

## 9. Toma de decisión

Los frentes consolidados de NSGA-II y MOPSO se combinaron en una sola matriz. De 344 soluciones iniciales se obtuvieron 336 alternativas diferentes; después de retirar 62 alternativas dominadas y agrupar soluciones con los mismos valores objetivo, la matriz final quedó compuesta por 272 pares únicos de ganancia y tiempo. Esta depuración evitó aplicar las técnicas multicriterio sobre soluciones que ya eran inferiores a otras.

### 9.1. TOPSIS

TOPSIS selecciona la alternativa más próxima a una solución ideal positiva y más alejada de una solución ideal negativa. La ganancia se normalizó como criterio de beneficio y el tiempo como criterio de costo:

$$
r_{iG}=\frac{G_i-G_{\min}}{G_{\max}-G_{\min}}
$$

$$
r_{iT}=\frac{T_{\max}-T_i}{T_{\max}-T_{\min}}
$$

Después se aplicaron los pesos \(w_G=w_T=0.50\), se calcularon las distancias euclidianas al ideal positivo y al ideal negativo, y se obtuvo el coeficiente de cercanía:

$$
C_i=\frac{D_i^-}{D_i^+ + D_i^-}
$$

La alternativa con mayor \(C_i\) fue seleccionada por TOPSIS.

### 9.2. Suma ponderada normalizada

La suma ponderada normalizada asigna a cada alternativa un puntaje agregado a partir de los criterios normalizados. Para este problema se calculó:

$$
S_i=0.50r_{iG}+0.50r_{iT}
$$

La alternativa con mayor \(S_i\) fue seleccionada como el mejor compromiso según esta técnica.

### 9.3. Resultados

| Técnica | \(x_1\) | \(x_2\) | \(x_3\) | Ganancia | Tiempo | Método de origen | Puntaje |
|---|---:|---:|---:|---:|---:|---|---:|
| TOPSIS | 0 | 48 | 0 | 3120 | 57,6 | MOPSO | 0,508885 |
| Suma ponderada normalizada | 0 | 30 | 0 | 1950 | 36,0 | MOPSO | 0,510255 |

TOPSIS seleccionó una alternativa intermedia, con mayor ganancia y mayor tiempo que la solución elegida por suma ponderada. La suma ponderada favoreció una alternativa más conservadora, cercana al mínimo de producción y con menor tiempo total. La diferencia se debe a que TOPSIS evalúa simultáneamente la proximidad al ideal positivo y el alejamiento del ideal negativo, mientras que la suma ponderada agrega directamente los dos desempeños normalizados.

### 9.4. Solución final recomendada

Se recomienda la solución seleccionada mediante TOPSIS:

$$
\mathbf{x}=[0,48,0]
$$

Esta alternativa propone producir 48 escritorios estándar, con una ganancia total de 3120 y un tiempo de producción de 57,6 horas. La elección se considera más equilibrada, puesto que evita concentrarse excesivamente en el menor tiempo y conserva una ganancia superior a la obtenida mediante suma ponderada.

### 9.5. Referencias para las técnicas multicriterio

[1] X. Liu, H. Guo, H. Chen, Y. Wu y D. Yu, “An Improved NSGA-II–TOPSIS Integrated Framework for Multi-Objective Optimization of Electric Vehicle Charging Station Siting,” *Sustainability*, vol. 18, art. 668, 2026, doi: 10.3390/su18020668.

[2] A. Ruiz-Vélez, J. García, J. Alcalá y V. Yepes, “Sustainable Road Infrastructure Decision-Making: Custom NSGA-II with Repair Operators for Multi-Objective Optimization,” *Mathematics*, vol. 12, no. 5, art. 730, 2024, doi: 10.3390/math12050730.
