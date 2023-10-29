// Input.tsx
import React from 'react';

interface IButtonProps {
  type: 'button' | 'submit' | 'reset';
  className: string;
  label: string;
  onClick?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const Button: React.FC<IButtonProps> = ({ type, className, label }) => {
  return (
    <button className={className} type={type}>
      {label}
    </button>
  );
};

export default Button;
