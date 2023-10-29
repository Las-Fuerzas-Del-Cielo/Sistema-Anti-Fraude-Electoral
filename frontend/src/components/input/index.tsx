import React, { useState } from "react";
import { IInputProps } from "./types";

const Input: React.FC<IInputProps> = ({
  label,
  type,
  id,
  placeholder,
  className,
  onChange,
}) => {
  const [isFocused, setIsFocused] = useState<boolean>(false);
  const [hasValue, setHasValue] = useState<boolean>(false);

  return (
    <div className="relative bg-white shadow-md rounded-xl font-sans w-full">
      <label
        htmlFor={id}
        className={`absolute left-4 top-1/3 transition-all duration-300 ease-in-out transform ${
          isFocused || hasValue
            ? "-translate-y-full text-xs"
            : "-translate-y-1/2 text-md"
        } pointer-events-none text-violet-brand`}
      >
        {label}
      </label>
      <input
        type={type}
        id={id}
        className={`block w-full bg-white border-custom-lightgray rounded-lg outline-none focus:border-violet-light placeholder-custom-lightgray pt-6 pb-2 h-16 font-normal ${className}`}
        placeholder={isFocused || hasValue ? "" : placeholder}
        onChange={(e) => {
          setHasValue(e.target.value !== "");
          if (onChange) {
            onChange(e);
          }
        }}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
      />
    </div>
  );
};

export default Input;
