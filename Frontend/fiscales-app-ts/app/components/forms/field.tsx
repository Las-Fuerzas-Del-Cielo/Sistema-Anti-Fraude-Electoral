"use client";

import React, { useState } from "react";

const DNIField = ({
  label = "Label",
  id = "43",
  name = "52",
  placeholder = "Ingresa tu DNI (Sin puntos ni comas)",
}) => {
  const [dni, setDni] = useState("");

  return (
    <div className="bg-white p-2 shadow-md rounded-md font-poppins">
      <label
        htmlFor="dni"
        className="block text-sm font-medium text-custom-purple mb-2"
      >
        {label}
      </label>
      <input
        type="text"
        id={id}
        name={name}
        placeholder={placeholder}
        value={dni}
        onChange={(e) => setDni(e.target.value)}
        className="border border-custom-blue p-2 w-full rounded-md text-custom-blue"
      />
    </div>
  );
};

export default DNIField;
