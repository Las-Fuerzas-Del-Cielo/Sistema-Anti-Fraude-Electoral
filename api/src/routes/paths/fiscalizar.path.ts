import { evaluateFiscalMesa } from '../controllers/fiscalizar'
import Route from "../../server/class/route";
import { MethodRoutes } from "../../enum/method.enum";

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
