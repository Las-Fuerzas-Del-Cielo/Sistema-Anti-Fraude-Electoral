import express, { json, urlencoded } from 'express'
import swaggerUi from 'swagger-ui-express'
import fs from 'fs-extra'
import Server from './server/server'

/**
 * @description Bootstrap the application with middlewares & swagger endpoint
 */
export function bootstrap() {
  const server =new Server({baseUrl:'/api'})
  
  // swagger
  if (!process.env.AWS_LAMBDA_FUNCTION_NAME) {
    server.setMiddleware('/api-docs', swaggerUi.serve, swaggerUi.setup(fs.readJsonSync('./swagger.json')))
  }

  return server
}
