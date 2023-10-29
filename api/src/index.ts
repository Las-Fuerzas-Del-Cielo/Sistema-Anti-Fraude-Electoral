require('dotenv').config()
import userRoutes from './routes/paths/user.path'
import mesaRoutes from './routes/paths/mesa.path'
import denunciaRoutes from './routes/denuncia'
import fiscalizarRoutes from './routes/paths/fiscalizar.path'
import { bootstrap } from './/bootstrap'
import mapRoute from './routes'

const server = bootstrap()

server.setRouteMap(mapRoute)
server.listen();
