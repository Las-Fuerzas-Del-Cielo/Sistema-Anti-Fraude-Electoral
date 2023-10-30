import { useNavigate } from 'react-router-dom';
import { useFormik } from 'formik';
import { observer } from 'mobx-react-lite';
import Button from '#/components/button';
import Input from '#/components/input';
import { ILoginProps } from './types';
import Footer from '#/components/footer';
import { Link } from 'react-router-dom';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();

  const onSubmit = (values: ILoginProps) => {
    //TODO: Add logic auth.
    //TODO: Change logic on submit with AUTH and redirect to /dashboard
    navigate('/dashboard');
  };

  const { handleSubmit, handleChange } = useFormik({
    initialValues: {
      dni: '',
      password: '',
    },
    onSubmit,
  });

  const handleClick = () => {
    // Maneja la lógica cuando se hace clic en el botón
  };

  return (
    <section className="relative flex flex-col items-center min-h-screen overflow-hidden bg-gray-100">
      <div className="z-10 w-5/6 p-4 md:w-1/2 shadow-3xl rounded-xl">
        <div className="container mx-auto">
          <div className="flex items-center justify-center my-20">
            <img
              src="src/assets/logos/fenix.png"
              alt="fenix"
              className="object-cover h-auto mr-4 rounded w-28"
            />
            <img
              src="src/assets/logos/lla.svg"
              alt="lla"
              className="object-cover h-auto rounded w-50"
            />
          </div>
        </div>
        <form className="w-full" onSubmit={handleSubmit}>
          <div className="flex items-center mb-6 text-lg md:mb-8 shadow-3xl">
            <Input
              label="DNI"
              type="text"
              id="dni"
              placeholder="Ingresa tu DNI"
              onChange={handleChange}
            />
          </div>
          <div className="flex items-center mb-6 text-lg md:mb-8 shadow-3xl">
            <Input
              label="Contraseña"
              type="password"
              id="password"
              placeholder="Ingresa tu Contraseña"
              onChange={handleChange}
            />
          </div>
          <div className="flex flex-col items-center text-lg">
            <Button
              className="w-full p-4 text-xl font-semibold tracking-wider text-white bg-violet-brand rounded-xl"
              type="submit"
              label="Ingresar"
              onClick={handleClick}
            />

            <Link to='total-results' className="mt-8 text-lg text-center text-gray-600 underline">
              Ir a resultados
            </Link>
          </div>
        </form>
      </div>
      <Footer />
    </section>
  );
};

export const Login = observer(LoginPage);

export default Login;
