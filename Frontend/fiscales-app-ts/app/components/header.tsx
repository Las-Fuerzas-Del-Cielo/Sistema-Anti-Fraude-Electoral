import Image from "next/image";
import DNIField from "./forms/field";

export default function Header() {
  return (
    <header className="fixed w-full bg-gray-800 p-4">
      <div className="mx-auto flex items-center justify-between">
        <div className="flex items-center">
          <Image src="/logo.png" alt="Logo" width={50} height={50} />
          <span className="ml-3 text-xl text-white font-semibold">
            La Libertad Avanza
          </span>
        </div>
        <nav>
          <ul className="flex space-x-4">
            <li>
              <a href="#" className="text-white">
                Inicio
              </a>
            </li>
            <li>
              <a href="#" className="text-white">
                Productos
              </a>
            </li>
            <li>
              <a href="#" className="text-white">
                Contacto
              </a>
            </li>
          </ul>
        </nav>
        <button className="bg-blue-500 text-white px-4 py-2 rounded-md">
          Iniciar Sesión
        </button>
      </div>
    </header>
  );
}
