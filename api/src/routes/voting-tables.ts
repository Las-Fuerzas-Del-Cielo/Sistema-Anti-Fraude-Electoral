import { Router } from 'express'
import { getVotingTableData, searchVotingTables, reportMissingAuditor } from '../controllers/voting-tables'

const router = Router()

router.get('/', searchVotingTables)
router.get('/:id', getVotingTableData)
router.post('/:id/report-missing-auditor', reportMissingAuditor)

export default router
