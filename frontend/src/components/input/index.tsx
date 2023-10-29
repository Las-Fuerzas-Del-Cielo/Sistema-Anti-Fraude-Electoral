import React, { useState } from "react";
import { IInputProps } from "./types";
import classNames from "classnames";

const Input: React.FC<IInputProps> = ({
  label,
  appearance,
  type,
  id,
  placeholder,
  
  className,
  labelClassName,
  inputClassName,

  onChange,
}) => {
  const outlineLabelApperence = 'border-2 relative rounded-xl bg-white shadow-md border-gray-300 focus-within:border-violet-light'
  const underlineLabelApperence = 'border-b-2 border-gray-300 focus-within:border-violet-light text-center'
  const labelApperence = appearance === 'underline' ? underlineLabelApperence : outlineLabelApperence

  const outlineSpanApperence = 'absolute top-1/3 transition-all transform duration-300 ease-in-out group-focus-within:-translate-y-full group-focus-within:text-sm -translate-y-1/2 pointer-events-none'
  const spanApperence = appearance === 'underline' ? '' : outlineSpanApperence

  const outlineInputApperence = 'pt-6 pb-1'
  const underlineInputApperence = 'text-center'
  const inputApperence = appearance === 'underline' ? underlineInputApperence : outlineInputApperence

  return (
    <div className="flex flex-col w-full gap-2 group">
      <label className={classNames("block font-sans w-full text-left px-3.5 py-2", labelApperence, className)}>
        <span className={classNames("text-md text-violet-brand", spanApperence, labelClassName)}>{label}</span>
        <input 
          id={id}
          type={type}
          placeholder={placeholder}
          onChange={onChange}
          className={classNames("w-full p-0 bg-transparent border-none focus:border-transparent focus:outline-none focus:ring-0 sm:text-sm", inputApperence, inputClassName)} />
      </label>
    </div>
  );
};

export default Input;
