import { Router } from 'express'
import { getSpecificDenuncia, listDenuncias } from '../controllers/denuncia'

const router = Router()

router.get('/', listDenuncias)

router.get('/:id', getSpecificDenuncia)

export default router
