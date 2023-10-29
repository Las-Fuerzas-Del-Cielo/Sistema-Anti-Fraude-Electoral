import { Router } from 'express'
import { createUser, getUserRoles, getUser } from '../controllers/user'

const router = Router()

router.post('/', createUser)
router.get('/:id/roles', getUserRoles)
router.get('/:id', getUser)

export default router
