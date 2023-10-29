import axios from 'axios'
import { RequestHandler } from 'express'
import { EleccionTipo, GetResultadosResponse, RecuentoTipo, ResultadosApi } from '../clients/resultadosApi'

export const getMesaData: RequestHandler = async (req, res) => {

  const {id: mesaId} = req.params;
  const {anioEleccion, tipoRecuento, tipoEleccion, categoriaId, distritoId, circuitoId, seccionId, seccionProvincialId } = req.query;

  await new ResultadosApi().getResultados({
    anioEleccion,
    tipoRecuento,
    tipoEleccion,
    categoriaId,
    distritoId,
    circuitoId,
    mesaId,
    seccionId,
    seccionProvincialId
  })
  .then( (response: GetResultadosResponse) => {
    return res.status(200).json(response);
  })
  .catch(error => {
      // La solicitud se completó, pero el servidor devolvió un código de estado distinto de 2xx (como 404, 500, etc.)
      res.status(error.response.status).send(error.response.data);
    
  });

}

export const searchMesas: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesas: ['Mesa 1', 'Mesa 2'] })
}
