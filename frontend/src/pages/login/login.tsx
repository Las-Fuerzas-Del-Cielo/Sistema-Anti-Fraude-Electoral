import Button from '#/components/button';
import Input from '#/components/input';
import { useFormik } from 'formik';
import { observer } from 'mobx-react-lite';
import { useNavigate } from 'react-router-dom';
import { ILoginProps } from './types';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();

  // eslint-disable-next-line no-unused-vars, @typescript-eslint/no-unused-vars
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

  return (
    <section className='relative flex flex-col items-center h-screen overflow-hidden bg-gray-100'>
      <div className='z-10 w-5/6 p-4 md:w-1/2 shadow-3xl rounded-xl'>
        <div className='container mx-auto'>
          <div className='flex items-center justify-center my-20'>
            <img alt='fenix' className='object-cover h-auto mr-4 rounded w-28' src='src/assets/logos/fenix.png' />
            <img alt='lla' className='object-cover h-auto rounded w-50' src='src/assets/logos/lla.svg' />
          </div>
        </div>
        <form className='w-full' onSubmit={handleSubmit}>
          <div className='flex items-center mb-6 text-lg md:mb-8 shadow-3xl'>
            <Input id='dni' label='DNI' placeholder='Ingresa tu DNI' type='text' onChange={handleChange} />
          </div>
          <div className='flex items-center mb-6 text-lg md:mb-8 shadow-3xl'>
            <Input
              id='password'
              label='Contraseña'
              placeholder='Ingresa tu Contraseña'
              type='password'
              onChange={handleChange}
            />
          </div>
          <div className='flex items-center mb-6 text-lg md:mb-8 shadow-3xl'>
            <Input
              id='password'
              label='Contraseña'
              placeholder='Ingresa tu Contraseña'
              type='password'
              onChange={handleChange}
            />
          </div>
          <div className='flex flex-col items-center text-lg'>
            <Button
              className='w-full p-4 text-xl font-semibold tracking-wider text-white bg-violet-brand rounded-xl'
              label='Ingresar'
              type='submit'
            />
            <a className='mt-8 text-lg text-center text-gray-600 underline' href='#'>
              ¿Necesitas ayuda?
            </a>
          </div>
        </form>
      </div>

      <div className='absolute left-0 right-0 transform -skew-y-12 -bottom-32 h-80 bg-violet-brand' />

      {/* 
        // TODO: FIX FOOTER IMAGE DESIGN 
        // https://www.figma.com/file/iO7j93Rxbk2nIfYdqpAmv2/%F0%9F%A6%85-APP-Fiscalizaci%C3%B3n-Libertaria-%7C-%F0%9F%93%B1-FINAL?type=design&node-id=59-4193&mode=dev
        <div className='flex flex-col items-center h-screen mt-auto overflow-hidden bg-gray-100 md:hidden'> <img /
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
