import { RequestHandler } from 'express';

export const listReports: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ denuncias: ['Denuncia 1', 'Denuncia 2'] });
};

export const getSpecificReport: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ denuncia: 'Specific denuncia data' });
};
