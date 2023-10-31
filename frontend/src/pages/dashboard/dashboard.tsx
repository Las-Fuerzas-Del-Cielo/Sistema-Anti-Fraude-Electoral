import { observer } from 'mobx-react-lite';
import { Link } from 'react-router-dom';
import Button from '#/components/button';
import Navbar from '#/components/navbar';
import { useAuth } from '#/context/AuthContext';

const DashboardPage = () => {
  const { user } = useAuth();
  return (
    <div className="min-h-screen">
      <Navbar />
      <section className="flex justify-center">
        <div className="md:w-1/2 w-5/6 shadow-3xl rounded-xl py-16 flex flex-col items-center space-y-8">
          <div className="container mx-auto flex flex-col items-center text-gray bg-white rounded-xl p-2 text-gray-dark shadow-md mb-12">
            <h2 className="text-xl font-bold mb-2">
              ¡Bienvenido {user?.firstName} {user?.lastName}!
            </h2>
            <p className="text-gray-500">{user?.email}</p>
          </div>
          <div className="flex flex-col items-center space-y-8 w-full">
            <Link to="/upload-certificate" className="w-full">
              <Button
                className="bg-violet-brand p-3 text-white w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light"
                type="submit"
                label="Cargar resultados de mesa"
              />
            </Link>
            <Link
              to="/total-results"
              className="border-2 border-violet-brand text-violet-brand bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light"
              type="submit"
            >
              Ver resultados
            </Link>
            <Button
              className="border-2 border-red text-red bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light"
              type="submit"
              label="Denunciar Irregularidades"
            />
            <div className="text-center text-gray-600 text-md mt-8">
              ¿No funciona el Formulario? <br />
              Realiza la denuncia{' '}
              <a href="#" className="text-violet-brand underline">
                aquí
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export const Dashboard = observer(DashboardPage);

export default Dashboard;
