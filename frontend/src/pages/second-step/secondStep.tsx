import { FC } from 'react';
import { Link } from 'react-router-dom';
import { observer } from 'mobx-react';
import Button from '#/components/button';

import './styles.css';
import ToggleButton from '#/components/toggleButton';

const SecondStep = () => {
  return (
    <section className='bg-gray-100 items-center flex flex-col '>
      <div className='bg-violet-brand p-4 w-full flex flex-col justify-center  items-center text-white '>
        <div className='w-full flex flex-row'>
          <div className='flex flex-col justify-center items-cente basis-1/8'>
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

        <h1 className='text-2xl font-semibold mb-5 mt-5 message'>Cargá el certificado fiscal</h1>
      </div>
      <div className='p-4'>
        <div className='container mx-auto'>
          <div className='flex items-center justify-center my-210'>
            <span>ProgressIndicator</span>
            {/* TODO: Add ProgressIndicator (FINISHED) <ProgressIndicator step_one='successful' step_two='successful'step_three='successful' step_four='successful' /> */}
          </div>
          <div className='flex items-center my-210 text-base my-5 w-305'>
            <div className='p-4'>
            
            <span>Chequeá que la imagen se vea nítida y completa antes de subirla</span>

            </div>
          </div>

          <div className='flex items-center justify-center my-10 '>
            <img
              src='src/assets/images/certfFiscal-test.png'
              alt='data sent successful'
              className='object-cover rounded w-68 h-auto'
            />
          </div>
          <div className="flex items-center text-sm my-10">
            <div className="flex items-center">
              <div className='p-10 '>
              <ToggleButton />

              </div>
              <div className='p-3'>
              <h3 className="text-xs p-3">
                Verifico que la imagen está firmada por el presidente de mesa y fue completado por mí previamente.
              </h3>
              </div>
            </div>
          </div>

          <div className='flex items-center justify-center'>
            {/* TODO: Mover a Dashboard */}
            <Link to='/'>
              <Button
                className='bg-violet-brand p-4 text-white w-full rounded-xl text-xl tracking-wider'
                type='submit'
                label='Enviar imagen'
              />
            </Link>
          </div>
          <div className='flex items-center justify-center my-10'>
            {/* TODO: Mover a Dashboard */}
            <Link to='/'>
              <Button
                className='bg-gray-200 p-4 text-white w-full rounded-xl text-xl tracking-wider'
                type='submit'
                label='Reintentar'
              />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
};

export const secondStep = observer(SecondStep);

export default SecondStep;
