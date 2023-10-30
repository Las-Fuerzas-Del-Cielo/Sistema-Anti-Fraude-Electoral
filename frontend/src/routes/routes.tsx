import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));
const LoadData = lazy(() => import('#/pages/load-data/loadData'));
const Dashboard = lazy(() => import('#/pages/dashboard/dashboard'));
const Inicio = lazy(() => import('#/pages/inicio/inicio'));
const UploadCertificate = lazy(
  () => import('#/pages/upload-certificate/uploadCertificate'),
);
const FilterPage = lazy(() => import('#/pages/results/filter'));

const AppRoutes: React.FC = () => (
  <Routes>
    <Route path="/login" element={<Login />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/profile" element={<Profile />} />
    <Route path="/send-success" element={<SendSuccess />} />
    <Route path="/load-data" element={<LoadData />} />
    <Route path="/upload-certificate" element={<UploadCertificate />} />
    <Route path="/inicio" element={<Inicio />} />
    <Route path="/" element={<Login />} />
    <Route path="/results" element={<FilterPage />} />
  </Routes>
);

export default AppRoutes;
