import { RequestHandler } from 'express'
import { registrarReporteEnS3 } from '../utils/s3Utils';
import { ERROR_CODES } from '../utils/errorConstants';
import { ReportFaltaFiscal, ResultadoRegistroS3 } from '../types/models';
import { generateUniqueId } from '../utils/generateUniqueId';
import { GetResultadosParamsRequest, GetResultadosResponse, ResultadosApi } from '../clients/resultadosApi'

// Define the expected structure of the request body
interface ReportarFaltaFiscalBody {
  fiscalId: string;
  escuelaId: string;
}

// Define the expected URL parameters
interface ReportarFaltaFiscalParams {
  id: string; // mesaId is received as 'id' in the URL
}

interface ValidatedQueryParams {
  anioEleccion?: string;
  tipoRecuento?: string;
  tipoEleccion?: string;
  categoriaId?: string;
  distritoId?: string;
  circuitoId?: string;
  seccionId?: string;
  seccionProvincialId?: string;
}

// Define a type for the valid query parameters after filtering
type ValidParams = Partial<ValidatedQueryParams>;

function getValidatedQueryParams(query: any): ValidatedQueryParams {
  return {
    anioEleccion: query.anioEleccion as string | undefined,
    tipoRecuento: query.tipoRecuento as string | undefined,
    tipoEleccion: query.tipoEleccion as string | undefined,
    categoriaId: query.categoriaId as string | undefined,
    distritoId: query.distritoId as string | undefined,
    circuitoId: query.circuitoId as string | undefined,
    seccionId: query.seccionId as string | undefined,
    seccionProvincialId: query.seccionProvincialId as string | undefined,
  };
}

export const getVotingTableData: RequestHandler = async (req, res) => {
  const { id: mesaId } = req.params;
  const queryParams: ValidatedQueryParams = getValidatedQueryParams(req.query);

  // Construir 'params' excluyendo las propiedades 'undefined'
  const params: Partial<GetResultadosParamsRequest> = {
    mesaId,
    ...Object.entries(queryParams)
      .filter(([_, value]) => value !== undefined) // Solo incluir propiedades definidas
      .reduce((obj, [key, value]) => {
        (obj as Partial<GetResultadosParamsRequest>)[key as keyof GetResultadosParamsRequest] = value;
        return obj;
      }, {})
  };

  try {
    // Hacer la llamada a la API
    const response: GetResultadosResponse = await new ResultadosApi().getResultados(params as GetResultadosParamsRequest);
    res.status(200).json(response);
  } catch (error) {
    // Verificar si 'error' es una instancia de Error y si 'response' existe
    if (error instanceof Error && 'response' in error && typeof error.response === 'object' && error.response !== null) {
      // Verificar si 'status' y 'data' existen en 'response'
      if ('status' in error.response && 'data' in error.response) {
        // Asegurarse de que 'status' sea un número
        const status = typeof error.response.status === 'number' ? error.response.status : 500;
        res.status(status).send(error.response.data);
      } else {
        res.status(500).json({ message: 'Error sin respuesta del servidor' });
      }
    } else {
      console.error('Error desconocido:', error);
      res.status(500).json({ message: 'Error interno del servidor' });
    }
  }
  
};

export const searchVotingTables: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesas: ['Mesa 1', 'Mesa 2'] })
} 

export const reportMissingAuditor: RequestHandler<ReportarFaltaFiscalParams, any, ReportarFaltaFiscalBody> = async (req, res) => {
  // Get mesaId from URL parameters
  const mesaId: string = req.params.id;

  // Destructure fiscalId and escuelaId from the request body
  const { fiscalId, escuelaId } : {fiscalId: string, escuelaId: string} = req.body;

  // Validación básica de los datos de entrada
  if (!fiscalId || !mesaId || !escuelaId) {
    return res.status(ERROR_CODES.INCOMPLETE_DATA.status).json({ message: ERROR_CODES.INCOMPLETE_DATA.message });
    
  }

  try {
    // Validar que el fiscal es un fiscal general
    const esFiscalGeneral: boolean = await validarFiscalGeneral(fiscalId);
    if (!esFiscalGeneral) {
      return res.status(ERROR_CODES.UNAUTHORIZED_GENERAL.status).json({ message: ERROR_CODES.UNAUTHORIZED_GENERAL.message });
    }

    // Detectar la institución del fiscal general
    const institucion: string = await getInstitucionDeFiscal(fiscalId);
    if (!institucion) {
      return res.status(ERROR_CODES.INSTITUTION_NOT_FOUND.status).json({ message: ERROR_CODES.INSTITUTION_NOT_FOUND.message });
    }

    // Validar que la mesa y la escuela existan y estén relacionadas
    const mesaValida: boolean = await validateMesaYEscuela(mesaId, escuelaId);
    if (!mesaValida) {
      return res.status(ERROR_CODES.INVALID_MESA_OR_ESCUELA.status).json({ message: ERROR_CODES.INVALID_MESA_OR_ESCUELA.message });
    }

    // Registrar en S3 el reporte de falta de fiscales
    const reporte: ReportFaltaFiscal = {
      id: generateUniqueId(), // Función que genera un ID único
      fiscalId,
      mesaId,
      escuelaId,
      timestamp: new Date(), // Asegúrate de que sea un objeto Date
      observaciones: '', // Puedes dejarlo vacío o agregar alguna observación por defecto
    };
    
    const resultadoS3: ResultadoRegistroS3 = await registrarReporteEnS3(reporte);

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

async function getInstitucionDeFiscal(fiscalId: string): Promise<string> {
  // Lógica para obtener la institución del fiscal
  // Ejemplo:
  // const fiscal = await FiscalModel.findById(fiscalId);
  // return fiscal ? fiscal.institucion : null;
  return 'Institución XYZ'; // Simulación, reemplazar con la lógica real
}

async function validateMesaYEscuela(mesaId: string, escuelaId: string): Promise<boolean> {
  // Lógica para validar que la mesa pertenece a la escuela
  // Ejemplo:
  // const mesa = await MesaModel.findById(mesaId);
  // return mesa && mesa.escuelaId === escuelaId;
  return true; // Simulación, reemplazar con la lógica real
}
