import { getCurrentYear } from '#/utils';

interface Props {}

const Footer: React.FC<Props> = () => {
  const currentYear = getCurrentYear();

  return (
    <footer className='mt-auto py-10 relative w-full h-36 sm:48 overflow-visible flex items-center justifty-center'>
      <div className='top-0 left-[-20vw] absolute w-[140vw] h-[50vh] bg-violet-brand sm:-rotate-6 -rotate-12'></div>
      <span className='sm:flex hidden z-10 mt-auto mx-auto text-white'>
        TODOS LOS DERECHOS RESERVADOS {currentYear} LLA NACIONAL
      </span>
    </footer>
  );
};

export default Footer;
