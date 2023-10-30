import React from 'react';
import { Link } from 'react-router-dom';
import { IFormHeaderProps } from './types';

// eslint-disable-next-line @typescript-eslint/no-unused-vars, no-unused-vars
const FormHeader: React.FC<IFormHeaderProps> = ({ routerLink, title }) => {
  return (
    <div className='bg-violet-brand p-4 w-full flex justify-between items-center text-white'>
      <div>
        <Link to={routerLink}>
          <img alt='Volver' className='object-cover rounded w-4 sm:w-8 h-auto' src='src/assets/images/back-arrow.svg' />
        </Link>
      </div>
      <div>
        <div className='flex-shrink-0'>
          <img alt='Logo' className='object-cover rounded w-12 h-12' src='src/assets/logos/fenix-new.svg' />
        </div>
      </div>
      <div>
        <img alt='Menú' className='cursor-pointer w-6 h-6' src='src/assets/images/menu.svg' />
      </div>
    </div>
  );
};

export default FormHeader;
