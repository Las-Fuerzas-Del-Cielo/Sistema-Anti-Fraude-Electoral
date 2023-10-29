import { createUser, getUserRoles, getUser, getUsers } from '../controllers/user'
import Route from "../../server/class/route";
import { MethodRoutes } from "../../enum/method.enum";

export default [
    new Route({
        path:'/',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:getUsers,
        active:true
    }),
    new Route({
        path:'/',
        method:MethodRoutes.POST,
        middlewares:[],
        auth:false,
        controller:createUser,
        active:true
    }),
    new Route({
        path:'/:id',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:getUser,
        active:true
    }),
    new Route({
        path:'/:id/roles',
        method:MethodRoutes.GET,
        middlewares:[],
        auth:false,
        controller:getUserRoles,
        active:true
    }),
]
