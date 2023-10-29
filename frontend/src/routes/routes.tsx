import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));
const LoadData = lazy(() => import('#/pages/load-data/loadData'));
const Dashboard = lazy(() => import('#/pages/dashboard/dashboard'));

const AppRoutes: React.FC = () => (
  <Routes>
    <Route element={<Login />} path='/login' />
    <Route element={<Dashboard />} path='/dashboard' />
    <Route element={<Profile />} path='/profile' />
    <Route element={<SendSuccess />} path='/send-success' />
    <Route element={<LoadData />} path='/load-data' />
    <Route element={<Login />} path='/' />
  </Routes>
);

export default AppRoutes;
