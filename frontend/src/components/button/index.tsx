import React from 'react';
import cs from 'classnames';
import { ButtonProps } from './types';

const Button: React.FC<ButtonProps> = ({ label, onClick, className, variant = 'default', type = 'button' }) => {
  const baseClasses =
    'w-full md:w-[calc(100% - 2rem)] h-12 rounded-md text-center font-medium transition-colors duration-300';
  const defaultClasses =
    'bg-violet-brand text-white hover:bg-violet-dark hover:border-violet-dark border-violet-brand text-lg';
  const secondaryClass =
    'bg-white text-purple-600 hover:bg-gray-light hover:border-gray-light border-violet-brand text-lg';
  const inactiveClasses = 'bg-gray-inactive text-text-off border-gray-inactive cursor-not-allowed text-lg';
  const warningClass = 'bg-white text-red hover-bg-gray-light hover-border-red border-red text-lg';
  const outlineClass = 'text-off text-base';

  const buttonClasses = cs(
    baseClasses,
    {
      [defaultClasses]: variant === 'default',
      [secondaryClass]: variant === 'secondary',
      [inactiveClasses]: variant === 'inactive',
      [warningClass]: variant === 'warning',
      [outlineClass]: variant === 'outline',
    },
    className
  );

  return (
    <button className={buttonClasses} disabled={variant === 'inactive'} type={type} onClick={onClick}>
      {label}
    </button>
  );
};

export default Button;
