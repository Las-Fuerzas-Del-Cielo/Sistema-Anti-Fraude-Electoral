from .config import settings
import requests


class ApiResultados:
    def __init__(self) -> None:
        self.url = settings.URL


    def _get(self,endpoint):
        return requests.get(self.url+endpoint).json()


    def get_resultados_generales(self):
        '''
            Obtiene los resultados de las elecciones generales

            Args: None

            Retrun : json
        '''

        endpoint = '/scope/data/getScopeData/00000000000000000000000b/1/1'
        return self._get(endpoint)


    def get_resultados_por_mesa(self, mesa_id):
        '''
            Obtiene los resultados segun el numero de mesa

            Args:
                mesa_id (str): id de la mesa de la cual se quiere obtener la foto del telgrama.

            Retrun : json
        '''

        endpoint=  '/scope/data/getScopeData/'
        return self._get(endpoint+mesa_id+'/1')


    def get_img_telegrama(self,mesa_id:str):
        '''
            Obtiene la foto del telegrama en formato tiff

            Args:
                mesa_id (str): id de la mesa de la cual se quiere obtener la foto del telgrama.

            Returns:
                json:
                    "encodingBinary": Binario de la imagen en base64
                    "fileName": Nombre definido para la imagen,
                    "imagenState": {
                        "state": NN,
                        "date": NN,
                    },
                    "bloqueoState": {
                        "state": NN,
                        "date": NN
                    },
                    "incidenciaState": NN,
                    "metadatos": {
                        "hash": NN,
                        "pollingStationCode": "id de la mesa",
                        "pages": [
                            {
                                "status": NN,
                                "scanningDate": NN,
                                "scanningUser": DNI EMISOR,
                                "pageNumber": NN,
                                "transmissionDate": Fecha emision,
                                "transmissionUserId": DNI EMISOR,
                                "transmissionUserName": NOMBRE EMISOR
                            }
                        ]
                    },
                    "hash": "NN",
                    "scopeId": NN
                }

            Nota : Los valores que aparece como NN son valores que no pude
                    interpretar a que hacen referencia

        '''

        endpoint = '/scope/data/getTiff/'
        return self._get(endpoint+mesa_id)




