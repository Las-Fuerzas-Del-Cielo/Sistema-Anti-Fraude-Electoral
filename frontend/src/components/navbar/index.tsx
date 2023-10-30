import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <div className="bg-violet-brand p-4 w-full flex flex-col h-18">
      <div className="w-full flex flex-row items-center justify-between">
        <div className="flex-shrink-0">
          <img
            src="/src/assets/logos/fenix-white.svg"
            alt="Logo"
            className="object-cover rounded w-16 h-16"
          />
        </div>
        <div className="flex-shrink-0">
          <Link to="/profile">
            <img
              src="/src/assets/icon/user.svg"
              alt="User profile"
              className="object-cover rounded w-8 h-8"
            />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
