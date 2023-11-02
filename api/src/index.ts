import { bootstrap } from './bootstrap';
import routes from './routes';
import config from './config';

const app = bootstrap()
const { port } = config

routes.forEach(({ prefix, router }) => app.use(prefix, router));

app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`));
