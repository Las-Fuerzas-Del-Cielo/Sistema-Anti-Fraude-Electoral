import React from 'react';
import { IButtonProps } from './types';

const Button: React.FC<IButtonProps> = ({ type, className, label }) => {
  return (
    <button className={className} type={type}>
      {label}
    </button>
  );
};

export default Button;
