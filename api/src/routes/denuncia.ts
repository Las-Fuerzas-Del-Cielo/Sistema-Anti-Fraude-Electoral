import { Router } from 'express'
import { getSpecificDenuncia, listDenuncias } from '../controllers/denuncia'
const router = Router()

router.get('/denuncia', listDenuncias)
router.get('/denuncia/:id', getSpecificDenuncia)

export default router
