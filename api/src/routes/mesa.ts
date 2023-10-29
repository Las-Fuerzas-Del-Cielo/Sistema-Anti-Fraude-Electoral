import { Router } from 'express'
import { getMesaData, searchMesas } from '../controllers/mesa'
const router = Router()

router.get('/mesa/:id', getMesaData)
router.get('/mesa', searchMesas)

export default router
