import { Response } from 'express'

export const sendResponse = (res: Response, statusCode: number, msg: string = '', data: any = {}) => {
  const isSuccessStatusCode = statusCode >= 200 && statusCode < 300

  // Si no hay mensaje y el código no es exitoso, intenta extraer el mensaje del error conocido
  if (!msg && !isSuccessStatusCode) {
    msg = isKnownError(data) ? data.message : 'An error occurred'
    console.error(msg, data)
  }

  res.status(statusCode).json({
    success: isSuccessStatusCode,
    msg,
    data: isSuccessStatusCode ? data : {}
  })
}

const isKnownError = (error: unknown): error is { message: string } => {
  return (error as { message: string }).message !== undefined
}
