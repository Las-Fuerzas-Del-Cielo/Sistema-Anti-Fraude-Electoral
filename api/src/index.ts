require('dotenv').config()
import userRoutes from './routes/user'
import mesaRoutes from './routes/mesa'
import denunciaRoutes from './routes/denuncia'
import fiscalizarRoutes from './routes/fiscalizar'
import { bootstrap } from './/bootstrap'
import express from 'express';

const app = bootstrap()
app.use( express.json() );

app.use('/api', userRoutes)
app.use('/api', mesaRoutes)
app.use('/api', denunciaRoutes)
app.use('/api', fiscalizarRoutes)


const port = process.env.PORT || 3000
app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`))
