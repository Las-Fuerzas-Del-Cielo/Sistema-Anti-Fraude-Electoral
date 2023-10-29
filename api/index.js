require('dotenv').config()
const express = require('express')
const swaggerUi = require('swagger-ui-express')
const YAML = require('yamljs')
const swaggerDocument = YAML.load('swagger.yaml')

const app = express()
const port = 3000

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument))

app.listen(port, () => {
  console.log(`Swagger UI running at http://localhost:${port}/api-docs`)
})
