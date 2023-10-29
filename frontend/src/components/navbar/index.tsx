import { Link } from 'react-router-dom'

const Navbar = () => {
  return (
    <div className='bg-violet-brand p-4 w-full flex flex-col '>
      <div className='w-full flex flex-row'>
        <div className='basis-auto w-full flex justify-start items-center'>
          <img src='src/assets/logos/fenix-white.svg' alt='User profile logo' className='object-cover rounded w-35 sm:w-36 lg:w-36 h-auto w-image-25' />
        </div>
        <div className='basis-auto w-full flex justify-end items-center'>
          <Link to='/profile'>
            <img src='src/assets/icon/user.svg' alt='User profile logo' className='object-cover rounded w-35 sm:w-36 lg:w-36 h-auto w-image-25' />
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Navbar
