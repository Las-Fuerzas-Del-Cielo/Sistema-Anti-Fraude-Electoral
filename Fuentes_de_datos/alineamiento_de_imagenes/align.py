# Por: Gissio
#
# Basado en:
# * https://github.com/Gissio/safe-ocr-alignment
# * https://github.com/FelixHertlein/inv3d-model

import os
import shutil
import sys

import inference

if len(sys.argv) != 3:
    print("Usage: python3 align.py [IMAGE-FILE] [TEMPLATE-FILE]")
    exit(1)

image_file = sys.argv[1]
template_file = sys.argv[2]

input_path = "input/elecciones/"

shutil.copyfile(image_file, input_path + "image_1.jpg")
shutil.copyfile(template_file, input_path + "template_1.jpg")

print(input_path)

inference.inference(
    model_name="geotr_template_large@inv3d",
    dataset="elecciones",
    output_shape=(1400, 860),
)

shutil.copyfile(
    "output/elecciones - geotr_template_large@inv3d/unwarped_1.png",
    "output/unwarped.png",
)
