import { Router } from 'express'
import { getSpecificReport, listReports } from '../controllers/report'


const router = Router()

router.get('', listReports)
router.get('/:id', getSpecificReport)

export default router
