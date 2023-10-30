import classNames from 'classnames';
import React, { useState } from 'react';
import { IInputProps } from './types';

const Input: React.FC<IInputProps> = ({
  label,
  appearance,
  type,
  id,
  placeholder,
  error,

  className,
  labelClassName,
  inputClassName,

  onChange,
}) => {
  const outlineLabelApperence: string =
    'border-2 relative rounded-xl bg-white shadow-md border-gray-300 focus-within:border-violet-light';
  const underlineLabelApperence: string = 'border-b-2 border-gray-300 focus-within:border-violet-light text-center';
  const labelApperence: string = appearance === 'underline' ? underlineLabelApperence : outlineLabelApperence;

  const outlineSpanApperence: string =
    'absolute top-1/3 transition-all transform duration-300 ease-in-out group-focus-within:-translate-y-full group-focus-within:text-sm -translate-y-1/2 pointer-events-none';
  const spanApperence: string = appearance === 'underline' ? '' : outlineSpanApperence;

  const outlineInputApperence: string = 'pt-6 pb-1';
  const underlineInputApperence: string = 'text-center';
  const inputApperence: string = appearance === 'underline' ? underlineInputApperence : outlineInputApperence;

  const [isPasswordVisible, setIsPasswordVisible] = useState<boolean>(false);

  return (
    <div className='flex flex-col w-full gap-2 group'>
      <div
        className={classNames(
          'block font-sans w-full text-left px-3.5 py-2',
          labelApperence,
          className,
          error && 'border-red'
        )}
      >
        <label
          className={classNames('text-md text-violet-brand', spanApperence, labelClassName, error && 'text-red')}
          htmlFor={id}
        >
          {label}
        </label>
        <div className='flex items-center gap-2'>
          <input
            className={classNames(
              'w-full p-0 bg-transparent border-none focus:border-transparent focus:outline-none focus:ring-0 sm:text-sm',
              inputApperence,
              inputClassName,
              error && 'text-red'
            )}
            id={id}
            placeholder={placeholder}
            type={type === 'password' && isPasswordVisible ? 'text' : type}
            onChange={onChange}
          />
          {type === 'password' &&
            (isPasswordVisible ? (
              <img
                alt='Show password'
                className='object-cover h-auto rounded cursor-pointer'
                src='src/assets/icon/eye.svg'
                onClick={() => setIsPasswordVisible(false)}
              />
            ) : (
              <img
                alt='Hide password'
                className='object-cover h-auto rounded cursor-pointer'
                src='src/assets/icon/eye-off.svg'
                onClick={() => setIsPasswordVisible(true)}
              />
            ))}
        </div>
      </div>
    </div>
  );
};

export default Input;
