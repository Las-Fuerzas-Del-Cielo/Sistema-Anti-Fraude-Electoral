import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <div className='bg-violet-brand p-4 w-full flex flex-col h-18'>
      <div className='w-full flex flex-row items-center justify-between'>
        <div className='flex-shrink-0'>
          <img alt='Logo' className='object-cover rounded w-16 h-16' src='src/assets/logos/fenix-white.svg' />
        </div>
        <div className='flex-shrink-0'>
          <Link to='/profile'>
            <img alt='User profile' className='object-cover rounded w-8 h-8' src='src/assets/icon/user.svg' />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
