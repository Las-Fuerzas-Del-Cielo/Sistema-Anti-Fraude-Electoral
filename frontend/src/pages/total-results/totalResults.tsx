import Button from '#/components/button';
import Navbar from '#/components/navbar';
import { observer } from 'mobx-react';
import { Link } from 'react-router-dom';

const TotalResultsPage = () => {
  const percentages = [61.05, 38.95];
  const votes = ['16,482,688', '10,517,312'];

  return (
    <div className='bg-white h-screen flex flex-col'>
      <Navbar />
      <div className='flex flex-col p-4'>
        <p className='font-bold text-2xl text-gray-700 mt-5'>Resultados totales</p>
        <Link
          className='border-2 border-violet-brand text-violet-brand bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light my-4'
          to='/filter-results'
        >
          Seleccionar filtros
        </Link>
      </div>

      <div className='lg:px-60 px-3'>
        {
          //Card Javier, VLL
        }
        <div className='flex flex-col border rounded-2xl h-[50%]'>
          <div className='flex flex-col'>
            <div className='flex flex-row justify-between mb-1'>
              <img alt='' className='m-1 w-16 h-14' src='src/assets/logos/fenix.png' />
              <div className='flex flex-col items-end mr-5 mt-2'>
                <span className='text-[12px] text-[#64748B]'>{votes[0]} votes</span>
                <p className='font-bold uppercase text-[#61439D] '>{percentages[1]}%</p>
              </div>
            </div>
            <div className='ml-10 mb-5'>
              <div className='w-[95%] rounded-md h-2 bg-[#CBD5E1]'>
                <div className='h-full bg-[#61439D] rounded-l' style={{ width: `${percentages[0]}%` }} />
              </div>
              <p className='text-[13px] font-bold uppercase text-[#61439D] flex items-start'>La libertad Avanza</p>
              <p className='text-[12px] whitespace-nowrap uppercase text-[#64748B] flex items-start'>
                JAVIER GERARDO MILEI - VICTORIA VILLARUEL
              </p>
            </div>
          </div>
        </div>
        <div className='my-4' />
        {
          //Card Massa, que asco
        }
        <div className='flex flex-col border rounded-2xl h-[50%]'>
          <div className='flex flex-col'>
            <div className='flex flex-row justify-between mb-1'>
              <svg
                className='m-1 w-16 h-14'
                fill='none'
                viewBox='0 0 56 56'
                xmlns='http://www.w3.org/2000/svg'
                xmlnsXlink='http://www.w3.org/1999/xlink'
              >
                <rect fill='url(#pattern0)' height='56' width='56' />
                <defs>
                  <pattern height='1' id='pattern0' patternContentUnits='objectBoundingBox' width='1'>
                    <use transform='scale(0.00465116 0.00444444)' xlinkHref='#image0_10_4663' />
                  </pattern>
                </defs>
              </svg>
              <div className='flex flex-col items-end mr-5 mt-2'>
                <span className='text-[12px] text-[#64748B]'>{votes[1]} votes</span>
                <p className='font-bold uppercase text-[#61439D] '>{percentages[1]}%</p>
              </div>
            </div>
            <div className='ml-10 mb-5'>
              <div className='w-[95%] rounded-md h-2 bg-[#CBD5E1]'>
                <div className='h-full bg-[#61439D] rounded-l' style={{ width: `${percentages[1]}%` }} />
              </div>
              <p className='text-[13px] font-bold uppercase text-[#61439D] flex items-start'>Union por la patria</p>
              <p className='text-[12px] whitespace-nowrap uppercase text-[#64748B] flex items-start'>
                Sergio tomas massa - agustin rossi
              </p>
            </div>
          </div>
        </div>
      </div>
      <div className='flex flex-col px-8 lg:px-60 mt-10'>
        <div className='border border-t-1 opacity-70' />
        <div className='my-2'>
          <span className='text-[17px] text-[#64748B]'>Total de votes</span>
          <p className='text-[25px] font-bold uppercase text-[#61439D]'>27,000,000</p>
        </div>
        <div className='border border-t-1 opacity-70' />
        <div className='flex flex-row justify-between mt-2 px-3'>
          <div className='flex flex-col'>
            <span className='text-[17px] text-[#64748B]'>Mesas escrutadas</span>
            <p className='text-[25px] font-bold uppercase text-[#61439D]'>90.00%</p>
          </div>
          <div className='flex flex-col'>
            <span className='text-[17px] text-[#64748B]'>Participación</span>
            <p className='text-[25px] font-bold uppercase text-[#61439D]'>76.36%</p>
          </div>
        </div>
      </div>
      <div className='mt-4 p-4'>
        <Button
          className='border-2 border-rose-700 text-rose-700 bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light my-4'
          label='Alerta Irregularidades'
          type='button'
        />
      </div>
    </div>
  );
};

export const TotalResults = observer(TotalResultsPage);
export default TotalResults;
