import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from ApiResultados import ApiResultados
import base64

api_client = ApiResultados()


def descargar_img_telegrama(mesa_id):
    response = api_client.get_img_telegrama(mesa_id)
    binary_data = base64.b64decode(response['encodingBinary'])
    file_name = response["fileName"]
    with open(f'./{file_name}', "wb") as tiff_file:
        tiff_file.write(binary_data)


#ejemplo 
descargar_img_telegrama('0100501926X')