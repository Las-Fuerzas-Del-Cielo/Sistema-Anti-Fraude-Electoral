import { Router } from 'express'
import { evaluateFiscalMesa } from '../controllers/fiscalizar'
import Route from 'src/server/class/route'
const router = Router()

router.post('/fiscalizar', evaluateFiscalMesa)

export default [
    new Route({
        path:'/',
        method:MethodRoutes.POST,
        middlewares:[],
        auth:false,
        controller:evaluateFiscalMesa,
        active:true
    })
]
