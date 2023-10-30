import { Router } from 'express'
import { getVotingTableData, searchVotingTables, reportarFaltaFiscal } from '../controllers/voting-tables'

const router = Router()

router.get('/', searchVotingTables)
router.get('/:id', getVotingTableData)
router.post('/mesas/reportarFaltaFiscal', reportarFaltaFiscal)

export default router
