# Frontend Fiscales - Documentación

**Instalación**

Primero, clona el repositorio y navega hasta el directorio del proyecto:

git clone <URL_DEL_REPOSITORIO>
cd frontend-fiscales
npm install o yarn install

**Scripts Disponibles**

- `npm run dev`: Inicia el servidor de desarrollo de Vite.
- `npm run build`: Compila el proyecto para producción usando TypeScript y Vite.
- `npm run lint`: Ejecuta ESLint para analizar el código TypeScript y TypeScript React.
- `npm run preview`: Compila el proyecto y lo muestra en modo de vista previa.
- `npm run server`: Inicia el servidor JSON usando `db.json` como base de datos mock.


# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type aware lint rules:

- Configure the top-level `parserOptions` property like this:

```js
   parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json', './tsconfig.node.json'],
    tsconfigRootDir: __dirname,
   },
```

- Replace `plugin:@typescript-eslint/recommended` to `plugin:@typescript-eslint/recommended-type-checked` or `plugin:@typescript-eslint/strict-type-checked`
- Optionally add `plugin:@typescript-eslint/stylistic-type-checked`
- Install [eslint-plugin-react](https://github.com/jsx-eslint/eslint-plugin-react) and add `plugin:react/recommended` & `plugin:react/jsx-runtime` to the `extends` list
# lla-front
