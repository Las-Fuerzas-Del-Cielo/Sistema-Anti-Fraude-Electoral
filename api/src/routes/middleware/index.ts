import { Middleware } from 'express'

// example auth middleware
export const auth: Middleware = (req, res, next) => {
  const { session } = req

  if (session?.userId) {
    return res.status(401).json({ error: 'Not authorized' })
  }
  next()
}
