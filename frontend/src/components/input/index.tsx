import React from 'react';
import { IInputProps } from './types';

const Input: React.FC<IInputProps> = ({
  type,
  id,
  placeholder,
  className,
  onChange,
}) => {
  return (
    <input
      type={type}
      id={id}
      className={className}
      placeholder={placeholder}
      onChange={onChange}
    />
  );
};

export default Input;
