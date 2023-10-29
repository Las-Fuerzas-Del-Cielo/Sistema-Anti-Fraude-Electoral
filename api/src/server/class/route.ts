import { auth } from "src/routes/middleware";

class Route {
    path:string;
    method:string;
    auth:Boolean;
    controller:Function;
    active:Boolean;
    middlewares:Array<Function>;

    constructor({path,method,controller,active,middlewares,auth}){
        this.path=path,
        this.method=method,
        this.auth = auth ?? true;
        this.controller = controller;
        this.active = active ?? true;
        this.middlewares = this.catchErrorsMiddlewares(middlewares);
    }
    catchErrorsMiddlewares(middlewares:Array<Function> = []): Array<Function>{

        if(this.auth){
            middlewares = [auth,...middlewares];
        }
        const newMiddlewares = middlewares.map((middleware:Function) =>{
            if(typeof middleware == "function"){
                return this.createCatchErrors(middleware);
            }
            return middleware;
        });
        
        return newMiddlewares
    }
    createCatchErrors(middleware:Function):Function {
        const newFunction = async (req,res,next)=>{
            try {
                await middleware.call(req,next,res,req);
            } catch (error) {
                next(error);
            }
        }
        return newFunction
    }
}
export default Route