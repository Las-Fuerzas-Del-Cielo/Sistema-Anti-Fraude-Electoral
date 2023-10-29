import React from 'react';
import { Link } from 'react-router-dom';
import { IFormHeaderProps } from './types';

const FormHeader: React.FC<IFormHeaderProps> = ({ routerLink, title }) => {
  return (
    <div className='bg-violet-brand p-4 w-full flex flex-col justify-center  items-center text-white '>
      <div className='w-full flex flex-row'>
        <div className='flex flex-col justify-center  items-cente basis-1/4'>
          <Link to={routerLink}>
            <img
              src='src/assets/images/back-arrow.svg'
              alt='fenix logo'
              className='object-cover rounded w-6  sm:w-8  h-auto '
            />
          </Link>
        </div>
        <div className='basis-auto w-full flex justify-center  items-center'>
          <img
            src='src/assets/logos/fenix-white.svg'
            alt='fenix logo'
            className='object-cover rounded w-35 sm:w-36 lg:w-36  h-auto w-image-25 '
          />
        </div>
      </div>

      <h1 className='text-4xl mb-5 mt-5 message'>{title}</h1>
    </div>
  );
};

export default FormHeader;
