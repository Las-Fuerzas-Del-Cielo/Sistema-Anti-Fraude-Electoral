import express, { json, urlencoded } from 'express'
import swaggerUi from 'swagger-ui-express'
import fs from 'fs-extra'
import cors from 'cors'
/**
 * @description Bootstrap the application with middlewares & swagger endpoint
 */
export function bootstrap() {
  const app = express()
  app.use(cors())
  app.use(json()).use(urlencoded({ extended: true }))
  // swagger
  if (!process.env.AWS_LAMBDA_FUNCTION_NAME) {
    app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(fs.readJsonSync('./swagger.json')))
  }

  return app
}
