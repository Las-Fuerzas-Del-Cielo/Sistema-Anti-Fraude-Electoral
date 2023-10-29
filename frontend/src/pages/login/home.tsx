import { useStore } from '#/store';
import { useFormik } from 'formik';
import { observer } from 'mobx-react-lite';

interface ILoginProps {
  dni: string;
  password: string;
}

const LoginHomeContent = () => {
  const { uiStore } = useStore();

  const submitForm = (values: ILoginProps) => {
    console.log(values);
  };

  const { handleSubmit, handleChange } = useFormik({
    initialValues: {
      dni: '',
      password: '',
    },
    onSubmit: submitForm,
  });

  return (
    <div className='bg-gray-100 h-screen overflow-hidden items-center flex flex-col'>
      <div className='md:w-1/2 w-full shadow-3xl rounded-xl p-4'>
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
          <div className='flex items-center text-lg mb-6 md:mb-8'>
            <input
              type='text'
              id='dni'
              className='bg-gray-200 rounded-xl pl-12 py-4 focus:outline-none w-full font-semibold'
              placeholder='Ingresa tu DNI'
              onChange={handleChange}
            />
          </div>
          <div className='flex items-center text-lg mb-6 md:mb-8'>
            <input
              type='password'
              id='password'
              className='bg-gray-200 rounded-xl pl-12 py-4 focus:outline-none w-full font-semibold'
              placeholder='Ingresa tu Contraseña'
              onChange={handleChange}
            />
          </div>
          <div className='flex flex-col items-center text-lg'>
            <button
              className='bg-purple-700 p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider'
              type='submit'
            >
              Ingresar
            </button>
            <a
              href='#'
              className='text-center text-gray-600 text-lg font-normal underline mt-8'
            >
              ¿Necesitas ayuda?
            </a>
          </div>
        </form>
      s</div>
      <div className='mt-auto bg-gray-100 h-screen overflow-hidden items-center flex flex-col md:hidden'>
        <img
          src='src/assets/logos/footer.svg'
          alt='footer'
          className='w-full h-full p-0 m-0'
        />
      </div>

      <span className='bg-violet p-5 text-6xl text-white rounded-xl'>🦁 VLLC!</span>
    </div>
  );
};

export const LoginHome = observer(LoginHomeContent);
