import express, { json, urlencoded } from 'express'
import SAFE from 'src/errors/safe';
import catchErrors from './middlewares/catchErrors';
import CreateRoutes from './class/createRoute';

interface options{
    baseUrl?:String
}

class Server{
    app:express;
    port:String;
    baseUrl:String;
    version:String;
    dev:String;
    constructor(options?:options){
        this.app = express();
        this.port = process.env.PORT ?? '3000'; 
        this.baseUrl =  process.env.BASE_URL ?? options.baseUrl ??'/'
        this.version = process.env.VERSION ?? undefined;
        this.dev=process.env.DEV;
        this.middlewares();
        this.routes()
    }
    middlewares(){
        this.app.use(express.json()).use(urlencoded({ extended: true }));

    }
    routes(){
        this.app.get("/serviceStatus",(req,res) =>{
            let infoStatus = {
                "description": 'Sistema Anti Fraude Electoral',
                "pod_id": process.pid,
                "active_incidents": null,
                "version": this.version ?? "0.0.1"
            };
            res.status(200).json(infoStatus)
        });
    }
    setMiddleware(...middlewares){
        this.app.user(...middlewares)
    }
    listen(){
        this.setFinishRoutes();
        this.app.listen(this.port, ()=>{
            console.log(`Server up and running. \nSwagger UI running at http://localhost:${this.port}/api-docs`);
        })
    }
    setRoutes(routes:CreateRoutes){
        const route = routes.createRoutes();
        this.app.use("/",route);
    }
    setRouteMap(routes=new Map()){
        routes.forEach((value, key)=>{
            const workspaceRoutesClass = new CreateRoutes(key).addRoutes(value)
            this.setRoutes(workspaceRoutesClass)
        });
    }
    setFinishRoutes(){
        this.app.use((req, res)=>{
            if(this.dev){
                console.log(`No se encontró el base Path ${req.originalUrl},${req.method}`)
            }
            res.status(404).json(new SAFE().urlInvalido);
        });
        this.app.use(catchErrors);
    }
}

export default Server;