import { Suspense } from 'react';
import AppRoutes from './routes/routes';
import './App.css';

function App() {
  return (
    // TODO: Agregar un spinner de carga o algun mensaje mientras se carga la app.
    <Suspense fallback={<div>Cargando...</div>}>
      <AppRoutes />
    </Suspense>
  );
}

export default App;
