import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));

const AppRoutes: React.FC = () => (
  <Routes>
    <Route path='/login' element={<Login />} />
    <Route path='/profile' element={<Profile />} />
    <Route path='/send-success' element={<SendSuccess />} />
    <Route path='/' element={<Login />} />
  </Routes>
);

export default AppRoutes;
