class SAFE {
    message:String
    code:Number
    logMessage?:String
    
    constructor(msg?:string){
        this.message = '';
        this.code = 0;
        this.logMessage = msg ??'' 
    }
    toJSON():SAFE{
        delete this.logMessage;
        return this;
    }
    get jsonInvalido(){
        this.code=0;
        this.message="JSON inválido";
        return this
    }
    get errorAplicacion(){
        this.code = 1;
        this.message='Error de Aplicacion.';
        return this;
    }
    get urlInvalido(){
        this.code=2;
        this.message="URL inválido";
        return this
    }
    get sinPermisos(){
        this.code = 3;
        this.message='Sin permisos suficientes.';
        return this;
    }
    get tokenExpired(){
        this.code = 4;
        this.message='Token expired.';
        return this;
    }
    get objectoNoDisponible(){
        this.code = 5;
        this.message='Objecto no disponible';
        return this;
    }
    get usrPassInvalida(){
        this.code = 6;
        this.message='Usuario y/o password inválidos.'
        return this;
    }
}
export default SAFE;