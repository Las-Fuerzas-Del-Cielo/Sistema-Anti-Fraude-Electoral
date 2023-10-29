import { RequestHandler } from 'express'

export const createUser: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(201).json({ message: 'User created', user: req.body })
}

export const getUserRoles: RequestHandler = (req, res) => {
  // Mocked Logic
  const roles = ['admin', 'user']
  res.status(200).json({ roles })
}

export const getUser: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json({ user: 'John Doe' })
}
export const getUsers: RequestHandler = (req, res) => {
  // Mocked Logic
  res.status(200).json([{ userId: 1 }, { userId: 2 }])
}
