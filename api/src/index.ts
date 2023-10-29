require('dotenv').config()
import express from 'express'
import { bootstrap } from './bootstrap'
const app = express()
const port = process.env.PORT || 3000

bootstrap(app)

app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`))
