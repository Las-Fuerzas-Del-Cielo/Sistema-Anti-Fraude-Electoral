import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));

const AppRoutes: React.FC = () => (
  <Routes>
    <Route element={<Login />} path='/login' />
    <Route element={<Profile />} path='/profile' />
    <Route element={<SendSuccess />} path='/send-success' />
    <Route element={<Login />} path='/' />
  </Routes>
);

export default AppRoutes;
