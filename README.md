%%writefile README.md
# Post-Contenido 1 - Unidad 11

## Estudiante
Nombre: Yulian Andres Ortega Machado-1152485

## Descripción

Repositorio correspondiente al laboratorio de introducción a CUDA.

El objetivo es comparar la ejecución de algoritmos en CPU y GPU utilizando:

- Suma de vectores (vectorAdd)
- Multiplicación de matrices (matMul)

## Estructura

src/
capturas/

## Estado

Proyecto inicializado.

## Checkpoint 1 - Vector Addition con CUDA

Se implementó un kernel CUDA para realizar la suma de dos vectores de tamaño 2^24. La aplicación compara el tiempo de ejecución de una versión secuencial en CPU con una versión paralela ejecutada en GPU utilizando CUDA.

### Resultado obtenido

| Métrica | Valor |
|----------|----------:|
| CPU | 48.62 ms |
| GPU Kernel | 106.61 ms |
| Errores | 0 |

### Análisis

La ejecución confirmó que la implementación CUDA produce resultados correctos, ya que no se detectaron diferencias entre los resultados generados por CPU y GPU. Para esta prueba específica, el tiempo del kernel GPU fue superior al tiempo obtenido en CPU. Este comportamiento puede explicarse por los costos asociados al lanzamiento del kernel y a la gestión de la ejecución paralela, los cuales pueden ser significativos cuando la carga de trabajo no es suficientemente grande para aprovechar completamente el paralelismo de la GPU.

### Evidencia

![VectorAdd](capturas/vectorAdd.png)