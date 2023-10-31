import RouterAdapter from './routerAdapter'
import userRoutes from './user'
import votingTablesRoutes from './voting-tables'
import auditRoutes from './audit'
import reportRouter from './report';
import uploadRoutes from './upload'

const routers: RouterAdapter[] = [
    new RouterAdapter('reports', reportRouter),
    new RouterAdapter('user', userRoutes),
    new RouterAdapter('voting-tables', votingTablesRoutes),
    new RouterAdapter('audit', auditRoutes),
    new RouterAdapter('upload', uploadRoutes),
];

export default routers;