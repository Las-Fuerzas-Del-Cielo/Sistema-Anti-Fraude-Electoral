# Descripcion

Base de api en nodejs express, contiene un [swagger ui endpoint](http://localhost:3000/api-docs) con un open API spec generado a partir de la definición del MVP.

Ref: [MVP](https://docs.google.com/document/d/11F_YE7d1th6ORO_AVKZn9idoiMjSzepwSYcMBZONDt8)

# Requirements

[Node.js (v18+)](https://nodejs.org/en/download)
[Docker](https://www.docker.com/products/docker-desktop/)
[Yarn](https://yarnpkg.com/getting-started/install)

# Setup Instructions

Para el desarrollo local, el proyecto emula los servicios de AWS con localstack. Por lo que no tenes que instalar ni configurar AWS CLI; incluso las variables `.env` se popularan automáticamente para que solo tengas que concentrarte en el desarrollo :).

```bash
cd api && docker-compose up && make start
```

## AWS SDK Examples

Una vez que esté corriendo localstack, podes mirar ejemplos de peticiones con el SDK de AWS en [ejemplos](examples/index.js).

O simplemente corré `npm run test:examples` para verificar que todo funcione correctamente.

# Swagger UI

Corre automáticamente cuando arranca la app, en este [endpoint](http://localhost:3000/api-docs).

## Generate

Corré

```bash
yarn build:swagger
```
