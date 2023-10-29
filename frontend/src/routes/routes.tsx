import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('../pages/login/login'));
const Profile = lazy(() => import('../pages/profile/profile'));

const AppRoutes: React.FC = () => (
  <Routes>
    <Route path='/login' element={<Login />} />
    <Route path='/profile' element={<Profile />} />
    <Route path='/' element={<Login />} />
  </Routes>
);

export default AppRoutes;
