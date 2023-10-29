import React from 'react';
import { IInputProps } from './types';

const Input: React.FC<IInputProps> = ({ type, id, placeholder, className, onChange }) => {
  return (
    // TODO: Agregar diseño como en el figma: https://www.figma.com/file/iO7j93Rxbk2nIfYdqpAmv2/%F0%9F%A6%85-APP-Fiscalizaci%C3%B3n-Libertaria-%7C-%F0%9F%93%B1-FINAL?type=design&node-id=59-4193&mode=dev
    <input type={type} id={id} className={className} placeholder={placeholder} onChange={onChange} />
  );
};

export default Input;
