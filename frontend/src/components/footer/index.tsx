interface Props {}

const Footer: React.FC<Props> = () => {
  return (
    <footer className='mt-auto py-10 relative w-full h-48 overflow-visible flex items-center justifty-center'>
      <div className='bottom-[-120px] left-[-20vw] absolute w-[140vw] h-56 sm:h-64 bg-violet-brand sm:-rotate-6 -rotate-12'></div>
      <span className='sm:flex hidden z-10 mt-auto mx-auto text-white'>
        TODOS LOS DERECHOS RESERVADOS 2023 LLA NACIONAL
      </span>
    </footer>
  );
};

export default Footer;
