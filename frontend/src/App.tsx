import { Suspense } from 'react';
import AppRoutes from './routes/routes';
import './App.css';

function App() {
  return (
    <Suspense fallback={<div>Cargando...</div>}>
      <AppRoutes />
    </Suspense>
  );
}

export default App;
