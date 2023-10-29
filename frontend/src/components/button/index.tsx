import React from 'react';
import { IButtonProps } from './types';

const Button: React.FC<IButtonProps> = ({ type, className, label, onClick }) => (
  <button className={className} type={type} onClick={onClick}>
    {label}
  </button>
);

export default Button;
