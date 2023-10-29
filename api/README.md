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
cd api && make start
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
## CREAR UNA NUEVA RUTA

Para crear una nueva ruta hay que seguir los siguientes pasos:

- Agregar dentro dentro del map de routes/index.ts el basePath de la route y un array de routas
- Crear dentro de la carpeta routes/path un archivo con la terminacion .path.ts
- Dentro de este archivo hay que crear un array con las rutas a utilizar 
```js
    new Route({
        path:'/', /// path de la routa
        method:MethodRoutes.GET, /// enum con los diferentes methods
        middlewares:[], /// donde irian los middlewares
        auth:false, /// autenticación en caso de utilizarla 
        controller:()=>{}, /// controler
    })
```
- Una vez creado la ruta hay que crear los controladores y los middlewares
- Para crear un controlador si o si tiene que ser una función no puede ser de flecha para poder acceder al objeto this que en este caso es la request y para responder un objeto al usuario solamente hay que hacer uso de result() este se encargara de responder automáticamente en caso de querer responder 400 se usaría reject() 
```js
    export const listDenuncias:Function = function(result,reject) {
    // Mocked Logic
    try{
        return result({ denuncias: ['Denuncia 1', 'Denuncia 2'] })
        
    }catch{
        return reject({err:true})
    }
    };
```
- Para crear un middleware seguir la misma lógica que anterior pero esta recibe next y res  para hacer uso

```js
    export const listDenuncias:Function = function(next,res) {
    // Mocked Logic
        return next()
    };
```
