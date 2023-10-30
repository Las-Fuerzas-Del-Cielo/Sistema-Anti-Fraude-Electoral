import { Router } from 'express'
import { getMesaData, searchMesas, reportarFaltaFiscal } from '../controllers/mesa'

const router = Router()

router.get('/:id', getMesaData)
router.get('', searchMesas)
router.post('/mesas/reportarFaltaFiscal', reportarFaltaFiscal)

export default router
