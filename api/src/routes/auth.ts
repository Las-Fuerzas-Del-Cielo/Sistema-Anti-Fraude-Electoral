import { Router } from 'express'
import { signIn, signOut, refreshToken, signUp } from '../controllers/auth'

const router = Router()

router.post('/auth/sign-in', signIn)
router.post('/auth/sign-out', signOut)
router.post('/auth/sign-up', signUp)
router.post('/auth/refresh-token', refreshToken)

export default router
