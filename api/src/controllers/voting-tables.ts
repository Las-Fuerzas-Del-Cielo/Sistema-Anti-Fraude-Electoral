import { RequestHandler } from 'express'

export const getVotingTableData: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesaData: 'some mesa data' })
}

export const searchVotingTables: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ mesas: ['Mesa 1', 'Mesa 2'] })
}
