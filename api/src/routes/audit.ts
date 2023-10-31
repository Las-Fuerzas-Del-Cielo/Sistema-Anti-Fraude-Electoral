import { Router } from 'express'
import { auditFiscalVotingTable } from '../controllers/audit'


const router = Router()

router.post('', auditFiscalVotingTable)

export default router
