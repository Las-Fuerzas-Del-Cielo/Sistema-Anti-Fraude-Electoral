// Input.tsx
import React from 'react';

interface IInputProps {
  type: 'text' | 'password';
  id: string;
  placeholder: string;
  className: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

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
