export interface User {
  id: string
  roles: string[]
  whatsapp: string
  mesa_id: string
}

export interface Audit {
  imagen: string
  validado: boolean
  errores: boolean
  observaciones: string
}

export interface Session {
  userId: string
}

// Interfaz para representar una Mesa
export interface Mesa {
  id: string;
  numero: number;
  escuelaId: string; // Referencia a la escuela asociada
}

// Interfaz para representar una Escuela
export interface Escuela {
  id: string;
  nombre: string;
  direccion: string;
}

// Interfaz para el reporte de falta de fiscales
export interface ReportFaltaFiscal {
  id: string; // ID único para el reporte
  fiscalId: string; // ID del fiscal que reporta
  mesaId: string; // ID de la mesa donde falta el fiscal
  escuelaId: string; // ID de la escuela asociada a la mesa
  timestamp: Date; // Fecha y hora del reporte
  observaciones: string; // Observaciones adicionales si son necesarias
}

interface ResultadoSubidaS3 {
  key: string;
  bucket: string;
  status: string;
  resultadoMetadata: unknown; // Aquí puedes especificar un tipo más preciso si lo conoces
}

interface ErrorSubidaS3 {
  error: true;
  mensaje: string;
  detalles: string;
  codigoError: number;
}

// Tipo de retorno unificado que puede ser uno de los dos anteriores
export type ResultadoRegistroS3 = ResultadoSubidaS3 | ErrorSubidaS3;
