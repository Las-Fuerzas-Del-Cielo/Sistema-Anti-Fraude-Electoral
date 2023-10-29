require('dotenv').config()
import express from 'express'
import swaggerUi from 'swagger-ui-express'
import YAML from 'yamljs'
const swaggerDocument = YAML.load('swagger.yaml')

const app = express()
const port = 3000
app.use(express.json()).use(express.urlencoded({ extended: true }))

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument))

app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`))
