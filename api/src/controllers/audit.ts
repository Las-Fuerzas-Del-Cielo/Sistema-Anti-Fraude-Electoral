import {RequestHandler} from 'express';
import {insertAudit} from "../services/audit";
import {validateFiscalizar} from "../utils/ModelsUtils";

export const auditFiscalVotingTable: RequestHandler = async (req, res) => {
  const body = req.body
  if (!validateFiscalizar(body))
    return res.status(400).json({message: "The result should be a Fiscalizar object."})

  const result = await insertAudit(body)
  if (!result) return res.status(500).json({message: "Error occured while trying to save Fiscalizar."})
  res.status(201).json({ message: 'Fiscalization recorded' });
};