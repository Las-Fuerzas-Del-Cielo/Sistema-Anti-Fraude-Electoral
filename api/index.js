const express = require('express');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');
const swaggerDocument = YAML.load('swagger.yaml');
const helmet = require('helmet');
const cors = require('cors');
const Ddos = require('ddos');
const requestIp = require('request-ip');

const app = express();
const port = 3000;

// Enable CORS for all requests
app.use(cors());
// Secure Express app by setting various HTTP headers. Documentation: https://helmetjs.github.io/
app.use(helmet());
// Prevent DOS attacks
const ddos = new Ddos({ burst: 10, limit: 15 });
app.use(ddos.express);
// Enable request IP logging
app.use(requestIp.mw());

// Serve Swagger UI at /api-docs endpoint
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// Start the server
app.listen(port, () => {
  console.log(`Swagger UI running at http://localhost:${port}/api-docs`);
});