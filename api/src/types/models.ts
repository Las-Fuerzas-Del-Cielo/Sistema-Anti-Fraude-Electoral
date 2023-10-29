export interface User {
  id: string
  roles: string[]
  whatsapp: string
  mesa_id: string
}

export interface Fiscalizar {
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
export interface ReporteFaltaFiscal {
  id: string; // ID único para el reporte
  fiscalId: string; // ID del fiscal que reporta
  mesaId: string; // ID de la mesa donde falta el fiscal
  escuelaId: string; // ID de la escuela asociada a la mesa
  timestamp: Date; // Fecha y hora del reporte
  observaciones: string; // Observaciones adicionales si son necesarias
}