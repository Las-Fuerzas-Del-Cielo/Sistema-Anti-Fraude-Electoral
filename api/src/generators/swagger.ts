import swaggerAutogen from 'swagger-autogen'
import fs from 'fs'
import path from 'path'
import config from '../config'

const doc = {
  info: {
    title: 'LLA Fraud detection API',
    description: 'API for the LLA Fraud detection project'
  },
  host: `localhost:${config.port}`,
  basePath: '/api',
  consumes: ['application/json'],
  produces: ['application/json']
}

const routesDir = 'dist/src/routes'
const files = fs.readdirSync(path.resolve(process.cwd(), 'dist/src/routes'))

const routeFiles = files
  .filter(file => file.endsWith('.js') && file !== 'index.js')
  .map(file => path.join(routesDir, file))

const outputFile = path.resolve(process.cwd(), './swagger.json')

swaggerAutogen(outputFile, routeFiles, doc)
