import { Router } from 'express'
import { createUser, getUserRoles, getUser, getUsers } from '../controllers/user'
const router = Router()

router.post('', createUser)
router.get('', getUsers)
router.get('/:id', getUser)
router.get('/:id/roles', getUserRoles)

export default router
