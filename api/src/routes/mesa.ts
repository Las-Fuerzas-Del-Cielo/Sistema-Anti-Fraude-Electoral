import { Router } from 'express'
import { getMesaData, searchMesas, reportarFaltaFiscal } from '../controllers/mesa'
const router = Router()

router.get('/mesa/:id', getMesaData)
router.get('/mesa', searchMesas)
router.post('/mesas/reportarFaltaFiscal', reportarFaltaFiscal)

export default router
