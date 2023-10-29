import axios from 'axios'
import { RequestHandler } from 'express'

export const getMesaData: RequestHandler = async (req, res) => {
  // Mocked Logic

  const {id: mesaId} = req.params;

  const {anioEleccion, tipoRecuento, tipoEleccion, categoriaId, distritoId} = req.query;

  if(!anioEleccion || !tipoRecuento || !tipoEleccion || !categoriaId || !distritoId){
    return res.status(400).send("parametros obligatorios: anioEleccion, tipoRecuento, tipoEleccion, categoriaId, distritoId")
  }


  try {

    const resp = await axios.get(`https://resultados.mininterior.gob.ar/api/resultados/getResultados`, { params: {
      anioEleccion,
      tipoRecuento,
      tipoEleccion,
      categoriaId,
      distritoId,
      mesaId 
    } })
  
    const data = resp.data;

    if(data == null || data == undefined) return res.status(400).send({ mesaData : "No data"});
    if(!data.hasOwnProperty("estadoRecuento")) return res.status(400).send({ mesaData : "No data recuento"});
    
    const { cantidadElectores, cantidadVotantes } = data.estadoRecuento;
    if(cantidadElectores == 0 && cantidadVotantes == 0 ) return res.status(400).send({ mesaData : "Mesa con 0 electores y 0 votantes"});
    if(cantidadVotantes > cantidadElectores) return res.status(400).send({ fraude: "hay mayor cantidad de votantes que electores en esta mesa", mesaData : data});
    


    res.status(200).json({ mesaData: data })

    
  } catch (error) {
    res.status(500).send(error);
  }

}

export const searchMesas: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesas: ['Mesa 1', 'Mesa 2'] })
}
