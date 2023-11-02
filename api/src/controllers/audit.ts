import {RequestHandler} from 'express';
import {insertAudit} from '../services/audit';
import {validateAudit} from '../utils/ModelsUtils';

export const auditFiscalVotingTable: RequestHandler = async ({body}, res) => {
  if (!validateAudit(body)) {
    return res.status(400).json({message: 'The result should be a Fiscalizar object.'})
  }

  const result = await insertAudit(body)
  if (!result) {
    return res.status(500).json({message: 'Error occured while trying to save Fiscalizar.'})
  }
  res.status(201).json({ message: 'Fiscalization recorded' });
};