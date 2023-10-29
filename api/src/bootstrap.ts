import { registerRoutes } from './routes'
import { Express, json, urlencoded } from 'express'
import swaggerUi from 'swagger-ui-express'
import YAML from 'yamljs'

const swaggerDocument = YAML.load('swagger.yaml')

/**
 * @description Bootstrap the application with middlewares, routes & swagger endpoint
 */
export function bootstrap(app: Express) {
  registerRoutes(app)
    .use(json())
    .use(urlencoded({ extended: true }))

  // swagger
  if (!process.env.AWS_LAMBDA_FUNCTION_NAME) {
    app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument))
  }
}
