import { RequestHandler } from 'express'

export const getMesaData: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesaData: 'some mesa data' })
}

export const searchMesas: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesas: ['Mesa 1', 'Mesa 2'] })
}
