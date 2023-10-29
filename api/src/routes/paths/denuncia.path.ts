import Route from "../../server/class/route";
import { getSpecificDenuncia, listDenuncias } from "../controllers/denuncia";
import { MethodRoutes } from "../../enum/method.enum";

export default [
    new Route({
        path:'/',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:listDenuncias,
    }),
    new Route({
        path:'/:id',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:getSpecificDenuncia,
    }),
]