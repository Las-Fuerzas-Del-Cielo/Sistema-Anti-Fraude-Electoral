import { Suspense } from 'react';
import AppRoutes from './routes/routes';
import './App.css';
import { LoadingPage } from './pages/loading-page';

function App() {
  return (
    // TODO: Agregar un spinner de carga o algun mensaje mientras se carga la app.
    <Suspense fallback={<LoadingPage />}>
      <AppRoutes />
    </Suspense>
  );
}

export default App;
