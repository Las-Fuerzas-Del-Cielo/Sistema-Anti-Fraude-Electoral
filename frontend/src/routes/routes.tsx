import { LoadingPage } from '#/pages/loading-page';
import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));
const LoadInformation = lazy(
  () => import('#/pages/load-information/loadInformation'),
);
const Dashboard = lazy(() => import('#/pages/dashboard/dashboard'));
const UploadCertificate = lazy(
  () => import('#/pages/upload-certificate/uploadCertificate'),
);
const VerifyCertificate = lazy(
  () => import('#/pages/verify-certificate/verifyCertificate'),
);
const TotalResults = lazy(() => import('#/pages/total-results/totalResults'));
const FilterPage = lazy(() => import('#/pages/results/filter'));
const NotFound = lazy(() => import('#/pages/not-found/notFound'));
const DeskData = lazy(() => import('#/pages/desk-data/DeskData'));

const AppRoutes: React.FC = () => (
  <Routes>
    {/* Auth */}
    <Route path="/login" element={<Login />} />

    {/* Cuenta */}
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/profile" element={<Profile />} />

    {/* Steps Formulario */}
    <Route path="/upload-certificate" element={<UploadCertificate />} />
    <Route path="/verify-certificate" element={<VerifyCertificate />} />
    <Route path="/load-information" element={<LoadInformation />} />
    <Route path="/send-success" element={<SendSuccess />} />
    <Route path="/load-desk-data" element={<DeskData />} />

    {/* Filters & Results */}
    <Route path="/filter-results" element={<FilterPage />} />
    <Route path="/total-results" element={<TotalResults />} />

    {/* Utils */}
    <Route path="/loading-page" element={<LoadingPage />} />
    <Route path="/" element={<Login />} />

    {/* 404 Not found */}
    <Route path="*" element={<NotFound />} />
  </Routes>
);

export default AppRoutes;
