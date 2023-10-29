import { RequestHandler } from 'express'
import { registrarReporteEnS3 } from '../utils/s3Utils';
import { ERROR_CODES } from '../utils/errorConstants';
import { ReporteFaltaFiscal, Mesa, Escuela } from '../types/models';
import { generateUniqueId } from '../utils/generateUniqueId';

// Definir tipos específicos para los ID
type FiscalId = string;
type MesaId = string;
export const getMesaData: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesaData: 'some mesa data' })
}

export const searchMesas: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesas: ['Mesa 1', 'Mesa 2'] })
}

export const reportarFaltaFiscal: RequestHandler = async (req, res) => {
  const { fiscalId, mesaId, escuelaId } = req.body;

  // Validación básica de los datos de entrada
  if (!fiscalId || !mesaId || !escuelaId) {
    return res.status(ERROR_CODES.INCOMPLETE_DATA.status).json({ message: ERROR_CODES.INCOMPLETE_DATA.message });
    
  }

  try {
    // Validar que el fiscal es un fiscal general
    const esFiscalGeneral = await validarFiscalGeneral(fiscalId);
    if (!esFiscalGeneral) {
      return res.status(ERROR_CODES.UNAUTHORIZED_GENERAL.status).json({ message: ERROR_CODES.UNAUTHORIZED_GENERAL.message });
    }

    // Detectar la institución del fiscal general
    const institucion = await obtenerInstitucionDeFiscal(fiscalId);
    if (!institucion) {
      return res.status(ERROR_CODES.INSTITUTION_NOT_FOUND.status).json({ message: ERROR_CODES.INSTITUTION_NOT_FOUND.message });
    }

    // Validar que la mesa y la escuela existan y estén relacionadas
    const mesaValida: boolean = await validarMesaYEscuela(mesaId, escuelaId);
    if (!mesaValida) {
      return res.status(ERROR_CODES.INVALID_MESA_OR_ESCUELA.status).json({ message: ERROR_CODES.INVALID_MESA_OR_ESCUELA.message });
    }

    // Registrar en S3 el reporte de falta de fiscales
    const reporte: ReporteFaltaFiscal = {
      id: generateUniqueId(), // Función que genera un ID único
      fiscalId,
      mesaId,
      escuelaId,
      timestamp: new Date(), // Asegúrate de que sea un objeto Date
      observaciones: '', // Puedes dejarlo vacío o agregar alguna observación por defecto
    };
    
    const resultadoS3 = await registrarReporteEnS3(reporte);

    // Verifica si 'resultadoS3' es del tipo 'ErrorSubidaS3'
    if ('error' in resultadoS3 && resultadoS3.error) {
      // Manejar el caso de error
      res.status(ERROR_CODES.S3_UPLOAD_ERROR.status).json({ message: ERROR_CODES.S3_UPLOAD_ERROR.message, detalles: resultadoS3.detalles });
    } else {
      // Manejar el caso de éxito
      res.status(200).json({ message: 'Reporte de falta de fiscal recibido', resultadoS3 });
    }
  } catch (error) {
    console.error(error);
    res.status(ERROR_CODES.INTERNAL_SERVER_ERROR.status).json({ message: ERROR_CODES.INTERNAL_SERVER_ERROR.message });
  }
}

// Aquí irían las implementaciones de las funciones auxiliares
// validarFiscalGeneral, obtenerInstitucionDeFiscal, validarMesaYEscuela, registrarReporteEnS3

async function validarFiscalGeneral(fiscalId: string): Promise<boolean> {
  // Lógica para verificar en la base de datos si el fiscal es general
  // Ejemplo:
  // const fiscal = await FiscalModel.findById(fiscalId);
  // return fiscal && fiscal.tipo === 'general';
  return true; // Simulación, reemplazar con la lógica real
}

async function obtenerInstitucionDeFiscal(fiscalId: string): Promise<string|null> {
  // Lógica para obtener la institución del fiscal
  // Ejemplo:
  // const fiscal = await FiscalModel.findById(fiscalId);
  // return fiscal ? fiscal.institucion : null;
  return 'Institución XYZ'; // Simulación, reemplazar con la lógica real
}

async function validarMesaYEscuela(mesaId: string, escuelaId: string): Promise<boolean> {
  // Lógica para validar que la mesa pertenece a la escuela
  // Ejemplo:
  // const mesa = await MesaModel.findById(mesaId);
  // return mesa && mesa.escuelaId === escuelaId;
  return true; // Simulación, reemplazar con la lógica real
}
