require('dotenv').config()
import userRoutes from './routes/user'
import mesaRoutes from './routes/mesa'
import denunciaRoutes from './routes/denuncia'
import fiscalizarRoutes from './routes/fiscalizar'
import authRoutes from './routes/auth'
import { bootstrap } from './/bootstrap'

const app = bootstrap()

app.use('/api', userRoutes)
app.use('/api', mesaRoutes)
app.use('/api', denunciaRoutes)
app.use('/api', authRoutes)

const port = process.env.PORT || 3000
app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`))
