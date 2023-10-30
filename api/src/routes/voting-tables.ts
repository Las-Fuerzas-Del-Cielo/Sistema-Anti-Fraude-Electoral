import { Router } from 'express'
import { getVotingTableData, searchVotingTables } from '../controllers/voting-tables'
const router = Router()

router.get('/', searchVotingTables)
router.get('/:id', getVotingTableData)

export default router
