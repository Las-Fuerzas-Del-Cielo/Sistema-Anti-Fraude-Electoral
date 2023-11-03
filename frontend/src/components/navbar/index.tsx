import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const Navbar: React.FC = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="bg-violet-brand p-4 px-8 w-full flex flex-col h-18">
      <div className="w-full flex flex-row items-center justify-between">
        <div className="flex-shrink-0">
          <img
            src="/src/assets/logos/fenix-white.svg"
            alt="Logo"
            className="object-cover rounded w-16 h-16"
          />
        </div>
        <div className="flex flex-col justify-center">
          <div
            className="flex justify-center cursor-pointer transform transition-transform hover:scale-110"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            {!menuOpen ? (
              <img
                src="/src/assets/icon/menu.svg"
                alt="User profile"
                className="object-cover rounded w-8 h-8"
              />
            ) : (
              <img
                src="/src/assets/icon/close.svg"
                alt="User profile"
                className="object-cover rounded w-6 h-6 mr-1"
              />
            )}
          </div>
          {menuOpen && (
            <div className="absolute bg-white right-6 top-20 rounded-xl px-1 shadow-2xl z-[500]">
              <div className="absolute top-[-15px] right-12 w-0 h-0">
                <svg width="50" height="20">
                  <polygon points="25,0 0,50 50,50" fill="white" />
                </svg>
              </div>
              <div className="w-full text-left py-4 px-8 pt-6 border-b-2 border-gray-100 font-bold text-xl text-violet-brand">
                <span>Javier</span>
              </div>
              <div className="flex flex-col px-8 py-8 gap-y-10 md:gap-y-8 items-start text-lg text-[#363F45]">
                {/* El gris pactado no se parece al de figma */}
                <Link
                  to="/profile"
                  className="scale-100 transform transition-transform hover:scale-105"
                >
                  Mi cuenta
                </Link>
                <Link
                  to="/upload-certificate"
                  className="scale-100 transform transition-transform hover:scale-105"
                >
                  Cargar resultados de mesa
                </Link>
                <Link
                  to="/dashboard"
                  className="scale-100 transform transition-transform hover:scale-105"
                  onClick={() => alert('No existe la ruta aún')}
                >
                  Impugnar mesa
                </Link>
                <Link
                  to="/dashboard"
                  className="scale-100 transform transition-transform hover:scale-105"
                  onClick={() => alert('No existe la ruta aún')}
                >
                  Denunciar Irregularidades
                </Link>
                <Link
                  to="/total-results"
                  className="scale-100 transform transition-transform hover:scale-105"
                >
                  Ver resultados
                </Link>
              </div>
              <div className="flex w-full text-left py-7 white px-8 border-t-2 border-gray-100 ">
                <div className="flex gap-2 scale-95 transform transition-transform hover:scale-105">
                  <img
                    src="/src/assets/icon/log-out.svg"
                    alt="User profile"
                    className="object-cover rounded text-violet-brand"
                  />
                  <Link to="/login">Cerrar sesión</Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Navbar;
