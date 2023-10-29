require('dotenv').config()
import { bootstrap } from './bootstrap'
import mapRoute from './routes'

const server = bootstrap()

server.setRouteMap(mapRoute)
server.listen();
