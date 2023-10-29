import RouterAdapter from './routerAdapter';
import userRoutes from './user'
import mesaRoutes from './mesa'
import fiscalizarRoutes from './fiscalizar'
import denunciaRouter from './denuncia';

const routers: RouterAdapter[] = [
    new RouterAdapter('denuncias', denunciaRouter),
    new RouterAdapter('user', userRoutes),
    new RouterAdapter('mesa', mesaRoutes),
    new RouterAdapter('fiscalizar', fiscalizarRoutes),
];

export default routers;