import { FC } from 'react';
import { Link } from 'react-router-dom';
import { observer } from 'mobx-react';
import Button from '#/components/button';
import ProgressIndicator from '#/components/progressIndicator';
import FormHeader from '#/components/formHeader';
import { ProgressStepStatus } from '#/components/progressIndicator/types';


const SecondStep = () => {
  return (
    <section className='items-center flex flex-col '>
      <FormHeader routerLink="/" title="Cargá el certificado del fiscal" />
      <div className='p-4 w-full'>
        <div className='container mx-auto flex-column my-210'>
          <ProgressIndicator
            steps={[
              ProgressStepStatus.Active,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
            ]}
          />
          <div className='flex items-center justify-center my-210'>
            <div>
            </div>
            {/* TODO: Add ProgressIndicator (FINISHED) <ProgressIndicator step_one='successful' step_two='successful'step_three='successful' step_four='successful' /> */}
          </div>

          <div className=''>
            <div className='p-4 mx-10 text-start my-2 mx-12 text-xl font-bold'>

              <span>Cargá el certificado del fiscal.</span>
              {/* TODO: Pensar los espaciados y quizá el width de la img */}
            </div>
          </div>
          <div className='flex items-center my-210 text-lg-300 w-305'>
            <div className='p-4 mx-10 text-start mx-12 text-base'>
              <span>Chequeá que la imagen se vea nítida y completa antes de subirla.</span>
            </div>
          </div>

          <div className='flex items-center justify-center my-2'>
            <img
              src='src/assets/images/certfFiscal-test.png'
              alt='data sent successful'
              className='object-cover rounded w-100 h-auto'
            />
          </div>

          <div className="flex items-center text-sm my-10">
            <div className="flex items-center px-12">
              <div className='p-2 '>
                <div className="inline-flex items-center">
                  <label
                    className="relative flex items-center p-3 rounded-full cursor-pointer"
                    data-ripple-dark="true"
                  >
                    <input
                      id="login"
                      type="checkbox"
                      className="before:content[''] peer relative h-5 w-5 cursor-pointer appearance-none rounded-md border border-blue-gray-200 transition-all before:absolute before:top-2/4 before:left-2/4 before:block before:h-12 before:w-12 before:-translate-y-2/4 before:-translate-x-2/4 before:rounded-full before:bg-blue-gray-500 before:opacity-0 before:transition-opacity checked:border-violet-500 checked:bg-violet-500 checked:before:bg-violet-500 hover:before:opacity-10"
                    />
                    <div className="absolute text-white transition-opacity opacity-0 pointer-events-none top-2/4 left-2/4 -translate-y-2/4 -translate-x-2/4 peer-checked:opacity-100">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-3.5 w-3.5"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                        stroke="currentColor"
                        strokeWidth="1"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        ></path>
                      </svg>
                    </div>
                  </label>
                  <label
                    className="mt-px font-light text-gray-700 cursor-pointer select-none"
                  >
                  </label>
                </div>
              </div>
              <div className='p-3'>
                <h3 className="text-start">
                  Verifico que la imagen está firmada por el presidente de mesa y fue completado por mí previamente.
                </h3>
              </div>
            </div>
          </div>
          {/* TODO: Agregar lógica de documento a los botones*/}
          {/* TODO: Agregar lógica de documento al reintentar y habilitado/desahabilitado */}
          <div className='flex items-center justify-center w-full'>
            {/* TODO: Mover a Dashboard */}
            <Link to='/'>
            <Button
              className="border-2 border-violet-brand text-violet-brand bg-transparent px-20 py-2 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light"
              type="submit"
              label="Enviar imagen"
            />
            </Link>
          </div>

          <div className='flex items-center justify-center my-5 px-10'>
            {/* TODO: Mover a Dashboard */}
            <Link to='/'>
              <Button
                className='px-20 py-2 w-full rounded-xl text-xl tracking-wider border-2 border-gray-200 text-gray-300'
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
