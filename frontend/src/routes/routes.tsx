import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';


const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));
const LoadData = lazy(() => import('#/pages/load-data/loadData'));
const Dashboard = lazy(() => import('#/pages/dashboard/dashboard'));
const UploadCertificate = lazy(
  () => import('#/pages/upload-certificate/uploadCertificate'),
);
const SecondStep = lazy(() => import('#/pages/second-step/secondStep'))

const AppRoutes: React.FC = () => (
  <Routes>
    <Route path="/login" element={<Login />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/profile" element={<Profile />} />
    <Route path="/send-success" element={<SendSuccess />} />
    <Route path="/load-data" element={<LoadData />} />
    <Route path="/upload-certificate" element={<UploadCertificate />} />
    <Route path="/" element={<Login />} />
    <Route path="/second-step" element={ <SecondStep />}></Route>
  </Routes>
);

export default AppRoutes;
