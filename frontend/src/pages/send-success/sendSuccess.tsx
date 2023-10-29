import { FC } from 'react';
import { Link } from 'react-router-dom';
import { observer } from 'mobx-react';
import Button from '#/components/button';
import { ISendSuccessProps } from './types';

import './styles.css';

const SendSuccessPage: FC<ISendSuccessProps> = ({ message }) => {
  return (
    <section className='bg-gray-100 items-center flex flex-col '>
      <div className='bg-violet-brand p-4 w-full flex flex-col justify-center  items-center text-white '>
        <div className='w-full flex flex-row'>
          <div className='flex flex-col justify-center  items-cente basis-1/4'>
            <Link to='/'>
              <img
                src='src/assets/images/back-arrow.svg'
                alt='data sent successful'
                className='object-cover rounded w-6  sm:w-8  h-auto '
              />
            </Link>
          </div>
          <div className='basis-auto w-full flex justify-center  items-center'>
            <img
              src='src/assets/logos/fenix-white.svg'
              alt='data sent successful'
              className='object-cover rounded w-35 sm:w-36 lg:w-36  h-auto w-image-25 '
            />
          </div>
        </div>

        <h1 className='text-4xl mb-5 mt-5 message'>Datos enviados con éxito</h1>
      </div>
      <div className='p-4'>
        <div className='container mx-auto'>
          <div className='flex items-center justify-center my-210'>
            <span>ProgressIndicator</span>
            {/* TODO: Add ProgressIndicator (FINISHED) <ProgressIndicator step_one='successful' step_two='successful'step_three='successful' step_four='successful' /> */}
          </div>
          <div className='flex items-center justify-center my-20 '>
            <img
              src='src/assets/images/square-logo.svg'
              alt='data sent successful'
              className='object-cover rounded w-68 h-auto'
            />
          </div>
          <div className='flex items-center justify-center my-20'>
            <h3 className='successfull'>
              {message ?? '¡ MUCHAS GRACIAS POR FISCALIZAR !'}
            </h3>
          </div>
          <div className='flex items-center justify-center my-20'>
            {/* TODO: Mover a Dashboard */}
            <Link to='/'>
              <Button
                className='bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider'
                type='submit'
                label='Volver a inicio'
              />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
};

export const SendSuccess = observer(SendSuccessPage);

export default SendSuccess;
