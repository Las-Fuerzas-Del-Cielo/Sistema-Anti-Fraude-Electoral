import { RequestHandler } from 'express';

export const listDenuncias: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ denuncias: ['Denuncia 1', 'Denuncia 2'] });
};

export const getSpecificDenuncia: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ denuncia: 'Specific denuncia data' });
};
