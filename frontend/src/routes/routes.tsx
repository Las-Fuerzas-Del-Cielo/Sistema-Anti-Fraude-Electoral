import { LoadingPage } from '#/pages/loading-page';
import { lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('#/pages/login/login'));
const Profile = lazy(() => import('#/pages/profile/profile'));
const SendSuccess = lazy(() => import('#/pages/send-success/sendSuccess'));
const LoadInformation = lazy(() => import('#/pages/load-information/loadInformation'));
const Dashboard = lazy(() => import('#/pages/dashboard/dashboard'));
const UploadCertificate = lazy(() => import('#/pages/upload-certificate/uploadCertificate'));
const VerifyCertificate = lazy(() => import('#/pages/verify-certificate/verifyCertificate'));
const TotalResults = lazy(() => import('#/pages/total-results/totalResults'));
const FilterPage = lazy(() => import('#/pages/results/filter'));
const DeskData = lazy(() => import('#/pages/desk-data/DeskData'));

const AppRoutes: React.FC = () => (
  <Routes>
    {/* Auth */}
    <Route element={<Login />} path='/login' />

    {/* Cuenta */}
    <Route element={<Dashboard />} path='/dashboard' />
    <Route element={<Profile />} path='/profile' />

    {/* Steps Formulario */}
    <Route element={<UploadCertificate />} path='/upload-certificate' />
    <Route element={<VerifyCertificate />} path='/verify-certificate' />
    <Route element={<LoadInformation />} path='/load-information' />
    <Route element={<SendSuccess />} path='/send-success' />
    <Route element={<DeskData />} path='/load-desk-data' />

    {/* Filters & Results */}
    <Route element={<FilterPage />} path='/filter-results' />
    <Route element={<TotalResults />} path='/total-results' />

    {/* Utils */}
    <Route element={<LoadingPage />} path='/loading-page' />
    <Route element={<Login />} path='/' />
  </Routes>
);

export default AppRoutes;
