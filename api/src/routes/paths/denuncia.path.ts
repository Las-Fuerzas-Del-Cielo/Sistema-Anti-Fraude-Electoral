import Route from "src/server/class/route";
import { getSpecificDenuncia, listDenuncias } from "../controllers/denuncia";

export default [
    new Route({
        path:'/',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:listDenuncias,
        active:true
    }),
    new Route({
        path:'/:id',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:getSpecificDenuncia,
        active:true
    }),
]