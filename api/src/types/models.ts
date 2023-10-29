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
