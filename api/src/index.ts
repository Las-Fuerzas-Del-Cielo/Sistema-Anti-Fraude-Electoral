require('dotenv').config();

import routes from "./routes";
import { DEFAULT_PORT } from "./constants";
import { bootstrap } from './bootstrap';
import { rateLimiter } from "./middleware";

const app = bootstrap();

app.use(rateLimiter);

routes.forEach(({ prefix, router }) => app.use(prefix, router));

const port = process.env.PORT || DEFAULT_PORT;
app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`));
