# Análisis de Gases Disueltos en Transformadores con TDA y Gráficas Especializadas

Este repositorio contiene herramientas de análisis aplicadas a datos de gases disueltos en transformadores, usando técnicas de **análisis topológico de datos (TDA)**, **visualizaciones UMAP**, el **Triángulo de Duval** y más.

## Estructura del Repositorio

- `duaval.py`: Realiza el análisis gráfico en el Triángulo de Duval a partir de los gases combustibles (CH₄, C₂H₂, C₂H₄) de una unidad, permitiendo identificar modos de falla incipiente.
- `kapper.py`: Implementa el algoritmo KeplerMapper para crear un grafo topológico sobre los datos de gases, detectando patrones y agrupaciones en el espacio de características normalizado.
- `ventana_movil.py`: Evalúa la evolución temporal del sistema a través de una distancia de Wasserstein (H₀) aplicada a ventanas móviles sobre derivadas de las concentraciones de gases, visualizando cambios significativos.
- `persi_6.py`: Compara dos unidades mediante homología persistente y visualización en diagramas de persistencia y espacio UMAP. Calcula distancias Wasserstein entre las unidades para cada dimensión topológica.

## Requisitos

Instala las dependencias necesarias utilizando pip:

```bash
pip install pandas numpy matplotlib scikit-learn umap-learn kmapper ripser persim
```

> **Nota:** Para `duaval.py` es necesario un módulo externo llamado `duvals_triangle_plotter.py`, el cual debe contener la lógica para graficar en el Triángulo de Duval.

## Datos

Los scripts esperan archivos `.csv` en la carpeta `raw CSV export/` con separador `;` y encabezados en español como `"Hidrógeno (ppm)"`, `"Metano (ppm)"`, etc.  
Los scripts están preparados para analizar datos del año 2025.

## Uso

Ejecuta cualquier script con:

```bash
python nombre_del_script.py
```

Por ejemplo:

```bash
python duaval.py
```

Los resultados se guardan como imágenes (`.png`) o archivos HTML en el caso del grafo Mapper.

## Seguridad

Los scripts han sido limpiados para remover cualquier información sensible como direcciones IP, credenciales o tokens. Revisa los archivos antes de subir datos privados.

## Licencia

MIT License
