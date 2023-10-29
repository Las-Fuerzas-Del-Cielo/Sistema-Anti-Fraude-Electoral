import { Router } from 'express'
import { evaluateFiscalMesa } from '../controllers/fiscalizar'

const router = Router()

router.post('/', evaluateFiscalMesa)

export default router
