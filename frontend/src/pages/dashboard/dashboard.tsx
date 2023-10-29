import Button from '#/components/button';
import Navbar from '#/components/navbar';
import { observer } from 'mobx-react-lite';
import { Link } from 'react-router-dom';

const DashboardPage = () => {
  const user: string = 'Javier Gerardo';
  const email: string = 'Javo@gmail.com';
  const href: string = '#';

  return (
    <div className='min-h-screen'>
      <Navbar />
      <section className='flex justify-center'>
        <div className='md:w-1/2 w-5/6 shadow-3xl rounded-xl py-16 flex flex-col items-center space-y-8'>
          <div className='container mx-auto flex flex-col items-center text-gray bg-white rounded-xl p-2 text-gray-dark shadow-md mb-12'>
            <h2 className='text-xl font-bold mb-2'>¡Bienvenido {user}!</h2>
            <p className='text-gray-500'>{email}</p>
          </div>
          <div className='flex flex-col items-center space-y-8 w-full'>
            <Link className='w-full' to='/upload-certificate'>
              <Button
                className='bg-violet-brand p-3 text-white w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light'
                label='Cargar resultados de mesa'
                type='submit'
              />
            </Link>
            <Button
              className='border-2 border-violet-brand text-violet-brand bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light'
              label='Ver resultados'
              type='submit'
            />
            <Button
              className='border-2 border-red text-red bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light'
              label='Denunciar Irregularidades'
              type='submit'
            />
            <div className='text-center text-gray-600 text-md mt-8'>
              ¿No funciona el Formulario? <br />
              Realiza la denuncia{' '}
              <a className='text-violet-brand underline' href={href}>
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
