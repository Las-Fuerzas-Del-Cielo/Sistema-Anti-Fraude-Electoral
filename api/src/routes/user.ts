import { Router } from 'express'
import { createUser, getUserRoles, getUser, getUsers } from '../controllers/user'
const router = Router()

router.post('/user', createUser)
router.get('/user', getUsers)
router.get('/user/:id', getUser)
router.get('/user/:id/roles', getUserRoles)

export default router
