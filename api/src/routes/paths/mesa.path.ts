import { getMesaData, searchMesas } from '../controllers/mesa'
import Route from 'src/server/class/route'

export default [
    new Route({
        path:'/',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:searchMesas,
        active:true
    }),
    new Route({
        path:'/:id',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:getMesaData,
        active:true
    }),
]
