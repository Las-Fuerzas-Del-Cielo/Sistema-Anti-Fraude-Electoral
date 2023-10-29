import { RequestHandler } from 'express'
import { logging } from '../utils/logging'
import { sendResponse } from '../utils/response'
import { User } from '../types/models'

const NAMESPACE = 'Auth Controller'

export const signIn: RequestHandler = async (req, res) => {
  const ENDPOINT = 'SignIn Method'
  logging.info(NAMESPACE, ENDPOINT)
  try {
    const { dni, password } = req.body

    const userResult = {}

    if (!userResult) return res.status(404).json({ message: 'User not found' })

    return sendResponse(res, 200, 'Inicio sesion correctamente!', userResult)
  } catch (error) {
    logging.error(NAMESPACE, ENDPOINT, error)
    return sendResponse(res, 500, '', error)
  }
}

export const refreshToken: RequestHandler = async (req, res) => {
  logging.info(NAMESPACE, 'RefreshToken Method')
}

export const signOut: RequestHandler = async (req, res) => {
  logging.info(NAMESPACE, 'SignOut Method')
}

export const signUp: RequestHandler = async (req, res) => {
  const ENDPOINT = 'SignUp Method'
  logging.info(NAMESPACE, ENDPOINT)

  try {
    const { dni, password } = req.body
    const userResult = {}
    return sendResponse(res, 200, 'Usuario creado correctamente!', userResult)
  } catch (error) {
    logging.error(NAMESPACE, ENDPOINT, error)
    return sendResponse(res, 500, '', error)
  }
}
