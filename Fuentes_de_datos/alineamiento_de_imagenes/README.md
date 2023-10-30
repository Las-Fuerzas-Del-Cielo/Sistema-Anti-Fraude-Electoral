# Alineamiento de imagenes de comprobantes de fiscales y telegramas de mesa

Este script permite alinear automáticamente las fotografías de comprobantes de fiscales y telegramas de mesa, para facilitar el reconocimiento automático de caracteres.

Basado en https://github.com/FelixHertlein/inv3d-model. Más detalles en https://github.com/Gissio/safe-ocr-alignment.

## Uso

    python align.py [IMAGE-FILE] [TEMPLATE-FILE]

En el primer uso, descarga automáticamente el modelo de machine learning.

El archivo de salida es `output/unwarped.png`.

El tamaño del archivo de salida puede controlarse en `align.py`, en la línea:

    output_shape=(1400, 860),

## Ejemplos

    python align.py examples/00001.jpg examples/template1.png
    python align.py examples/00002.jpg examples/template1.png
    python align.py examples/00003.jpg examples/template1.png

## Tareas para hacer

* Ajustar el contraste automáticamente para mejorar la precisión.
* Optimizar la conversión por batches de imágenes.
* Reconocer automáticamente la plantilla.
