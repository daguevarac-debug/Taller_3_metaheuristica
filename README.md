# Taller 3 - Optimizacion multiobjetivo

Este repositorio contiene la implementacion modular de NSGA-II y MOPSO para
el problema de produccion de escritorios metalicos, junto con un notebook
reproducible que ejecuta el flujo completo del taller.

## Ejecucion del notebook

Instale las dependencias:

```bash
pip install -r requirements.txt
```

Ejecute el notebook de punta a punta con:

```bash
jupyter nbconvert --to notebook --execute Taller_3_Optimizacion_Multiobjetivo.ipynb --output Taller_3_Optimizacion_Multiobjetivo_ejecutado.ipynb
```

El notebook regenera los CSV y PNG dentro de `resultados/` desde las corridas
actuales de los algoritmos, sin depender de archivos generados previamente.
