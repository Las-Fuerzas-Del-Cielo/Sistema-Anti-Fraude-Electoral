import SAFE from "../../errors/safe";

export default (err, req, res, next) => {
    if(process.env.DEV){
        console.log(err)
    }
    if(err instanceof SAFE || err.constructor.name == 'SAFE'){
        if(err.logMessage != ''){
            console.log(err.logMessage);
        }
        return res.status(400).json(err)
    }        
    // if(err instanceof TokenExpiredError || err.constructor.name == 'TokenExpiredError'){
    //     return res.status(403).json(new SAFE().sinPermisos)
    // }
    // if(err instanceof JsonWebTokenError || err.constructor.name == 'JsonWebTokenError'){
    //     return res.status(403).json(new SAFE().sinPermisos)
    // }
    
    console.log(err,"Error general");
    return res.status(400).json(new SAFE().errorAplicacion);
}