import React, { useState } from "react";
import iso from "#/assets/images/whiteIso.svg";

const Navbar = () => {
  const [pageChanged, setPageChanged] = useState<boolean>(true);

  const handlePageChange = () => {
    pageChanged ?  window.location.href = "/" : console.error("Error de redirección");
    setPageChanged(!pageChanged);
  };
  return (
    <nav className="justify-content-center bg-[#61439D] m-0 p-2 w-full">
      <div className="flex items-center justify-between">
        {pageChanged ? (
          <div className="text-black text-2xl font-semibold p-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="1em"
              viewBox="0 0 448 512"
              fill="white"
              className="w-8 h-8 transition-transform transform hover:scale-90 duration-300 ease-in-out cursor-pointer"
              onClick={handlePageChange}
            >
              <path d="M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l160 160c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L109.2 288 416 288c17.7 0 32-14.3 32-32s-14.3-32-32-32l-306.7 0L214.6 118.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-160 160z" />
            </svg>
          </div>
        ) : null}
        <div className="text-white text-2xl font-semibold px-2 pb-1">
          <img
            src={iso}
            alt="LLA Logo"
            className="h-10 cursor-pointer transform transition-transform duration-500 ease-out hover:scale-125"
            onClick={handlePageChange}
          />
        </div>
        <div className="hidden text-white text-2xl font-semibold p-2"> {/* Está hidden porqué estamos en /profile */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="1em"
            viewBox="0 0 448 512"
            fill="white"
            className="cursor-pointer transform transition-transform duration-500 ease-out hover:scale-125"
          >
            <path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z" />
          </svg>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
