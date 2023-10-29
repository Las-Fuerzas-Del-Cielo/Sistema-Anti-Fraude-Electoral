import { Router } from 'express'
import { evaluateFiscalMesa } from '../controllers/fiscalizar'
const router = Router()

router.post('/fiscalizar', evaluateFiscalMesa)

export default router
