export interface User {
  id: string

  dni: string
  password: string

  firstName: string
  lastName: string
  
  email?: string
  phone?: string

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
