import { Router } from 'express'
import { getMesaData, searchMesas } from '../controllers/mesa'
const router = Router()

router.get('/:id', getMesaData)
router.get('', searchMesas)

export default router
