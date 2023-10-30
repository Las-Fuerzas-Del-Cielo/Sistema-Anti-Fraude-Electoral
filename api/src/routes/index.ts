import RouterAdapter from './routerAdapter'
import userRoutes from './user'
import mesaRoutes from './mesa'
import fiscalizarRoutes from './fiscalizar'
import uploadRoutes from './upload'
import denunciaRouter from './denuncia'


const routers: RouterAdapter[] = [
    new RouterAdapter('denuncia', denunciaRouter),
    new RouterAdapter('user', userRoutes),
    new RouterAdapter('mesa', mesaRoutes),
    new RouterAdapter('fiscalizar', fiscalizarRoutes),
    new RouterAdapter('upload', uploadRoutes),
];

export default routers;