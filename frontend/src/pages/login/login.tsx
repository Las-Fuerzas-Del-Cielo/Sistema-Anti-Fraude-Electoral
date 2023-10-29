import { useNavigate } from 'react-router-dom';
import { useFormik } from 'formik';
import { observer } from 'mobx-react-lite';
import Button from '#/components/button';
import Input from '#/components/input';
import { ILoginProps } from './types';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();

  const onSubmit = (values: ILoginProps) => {
    //TODO: Add logic auth.
    //TODO: Change logic on submit with AUTH and redirect to /dashboard
    navigate('/profile');
  };

  const { handleSubmit, handleChange } = useFormik({
    initialValues: {
      dni: '',
      password: '',
    },
    onSubmit,
  });

  return (
    <section className='bg-gray-100 h-screen overflow-hidden items-center flex flex-col relative'>
      <div className='md:w-1/2 w-5/6 shadow-3xl rounded-xl p-4 z-10'>
        <div className='container mx-auto'>
          <div className='flex items-center justify-center my-20'>
            <img
              src='src/assets/logos/fenix.png'
              alt='fenix'
              className='object-cover rounded w-28 h-auto mr-4'
            />
            <img
              src='src/assets/logos/lla.svg'
              alt='lla'
              className='object-cover rounded w-50 h-auto'
            />
          </div>
        </div>
        <form className='w-full' onSubmit={handleSubmit}>
          <div className='flex items-center text-lg mb-6 md:mb-8 shadow-3xl'>
            <Input
              label='DNI'
              type='text'
              id='dni'
              className='bg-gray-200 rounded-xl pl-4 py-4 focus:outline-none w-full border-2 border-gray-300'
              placeholder='Ingresa tu DNI'
              onChange={handleChange}
            />
          </div>
          <div className='flex items-center text-lg mb-6 md:mb-8 shadow-3xl'>
            <Input
              label='Contraseña'
              type='password'
              id='password'
              className='bg-gray-200 rounded-xl pl-4 py-4 focus:outline-none w-full border-2 border-gray-300'
              placeholder='Ingresa tu Contraseña'
              onChange={handleChange}
            />
          </div>
          <div className='flex flex-col items-center text-lg'>
            <Button
              className='bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider'
              type='submit'
              label='Ingresar'
            />
            <a
              href='#'
              className='text-center text-gray-600 text-lg underline mt-8'
            >
              ¿Necesitas ayuda?
            </a>
          </div>
        </form>
      </div>

      <div className='absolute -bottom-32 left-0 right-0 h-80 bg-violet-brand transform -skew-y-12'></div>

      {/* 
        // TODO: FIX FOOTER IMAGE DESIGN 
        // https://www.figma.com/file/iO7j93Rxbk2nIfYdqpAmv2/%F0%9F%A6%85-APP-Fiscalizaci%C3%B3n-Libertaria-%7C-%F0%9F%93%B1-FINAL?type=design&node-id=59-4193&mode=dev
        <div className='mt-auto bg-gray-100 h-screen overflow-hidden items-center flex flex-col md:hidden'> <img /
            src='src/assets/logos/footer.svg'
            alt='footer'
            className='w-full h-full p-0 m-0'
          /> 
        </div>
      */}
    </section>
  );
};

export const Login = observer(LoginPage);

export default Login;
