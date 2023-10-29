import DataSuccessful from '#/components/dataSentSuccessful';
import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('../pages/login/login'));

const AppRoutes: React.FC = () => (
  <Routes>
    <Route path='/login' element={<Login />} />
    <Route path='/' element={<Login />} />
    <Route path='/envio-existoso' element={<DataSuccessful />} />
  </Routes>
);

export default AppRoutes;
