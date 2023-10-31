import React from 'react';
import { Link } from 'react-router-dom';
import { IFormHeaderProps } from './types';

const FormHeader: React.FC<IFormHeaderProps> = ({ routerLink, title }) => {
  return (
    <div className='bg-violet-brand p-4 w-full flex justify-between items-center text-white'>
      <div>
        <Link to={routerLink}>
          <img
            src='src/assets/images/back-arrow.svg'
            alt='Volver'
            className='object-cover rounded w-4 sm:w-8 h-auto'
          />
        </Link>
      </div>
      <div>
      <div className="flex-shrink-0">
          <img
            src="src/assets/logos/fenix-new.svg"
            alt="Logo"
            className="object-cover rounded w-12 h-12"
          />
        </div>
      </div>
      <div>
        <img
          src='src/assets/images/menu.svg'
          alt='Menú'
          className='cursor-pointer w-6 h-6'
        />
      </div>
    </div>

  );
};

export default FormHeader;
