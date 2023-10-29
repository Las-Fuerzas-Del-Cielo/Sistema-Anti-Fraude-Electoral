import { Router } from 'express'
import Route from './route';
import SAFE from "../../errors/safe";

interface options{
    auth:boolean
}

class CreateRoutes{
    router:Router
    path:String
    Routes:Array<Route>
    options:Object
    auth:Boolean
    constructor(path:String,options?:options){
        this.router = Router();
        this.path=path
        this.Routes=[]
        this.options =options;
        this.auth = options?.auth;
    }
    addRoutes(routes:Array<Route>):CreateRoutes{
        routes = routes.flat();
        routes.forEach((route:Route) =>{
            if(route instanceof Route){
                return this.Routes.push(route);
            }
        })
        return this
    }
    createRoutes():Router{
        this.Routes.forEach(route=>{
            if(!route.active){
                return;
            }            
            const fullPath = `${this.path}${route.path}`;            
            this.router[route.method](fullPath,route.middlewares,async(req,res,next)=>{
                try {
                    const controller = route.controller;
                    await controller.call(req,async(result)=>{
                        if(typeof result == 'object'){
                            res.body = result;
                            res.status(200).json(result);
                        }else{
                            console.log(`Error al enviar al result un dato del tipo equivocado: ${result}`);
                            let err = new SAFE('Objeto invalido').errorAplicacion;
                            return next(err)
                        }
                    },(reject)=>{
                        if(typeof reject == 'object'){
                            return res.status(400).json(reject)
                        }else{
                            console.log(`Error al enviar al result un dato del tipo equivocado: ${reject}`);
                            let err = new SAFE('Objeto invalido').errorAplicacion;
                            return next(err)
                        }
                    });
                } catch (error) {
                    next(error)
                } 
            });
        })
        return this.router
    }
}
export default CreateRoutes