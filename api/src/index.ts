require('dotenv').config()

import { bootstrap } from './bootstrap'
import routes from "./routes";

const app = bootstrap()

routes.forEach(({ prefix, router }) => app.use(prefix, router));

const port = process.env.PORT || 3000
app.listen(port, () => console.log(`Server up and running. \nSwagger UI running at http://localhost:${port}/api-docs`))
