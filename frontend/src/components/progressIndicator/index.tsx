import './styles.css';
import { IProgressIndicatorProps, ProgressStepStatus } from './types';

const ProgressIndicator = ({ steps }: IProgressIndicatorProps) => {
  return (
    <div className='w-full flex justify-between items-center px-20 mt-4'>
      {steps.map((step, index) => (
        <>
          <div
            key={index}
            className={`circle flex justify-center items-center rounded-full ${
              step === ProgressStepStatus.Active
                ? 'bg-violet-brand text-white'
                : step === ProgressStepStatus.Successful
                ? 'bg-green-check text-white'
                : 'bg-gray-light text-black'
            }`}
          >
            {step === ProgressStepStatus.Successful ? (
              <img alt='' className='w-4 h-4' src='src/assets/check-icono.svg' />
            ) : (
              <span className='font-normal text-xl'>{(index + 1).toString()}</span>
            )}
          </div>

          {index != steps.length - 1 && (
            <div
              className={`stick ${
                step === ProgressStepStatus.Active
                  ? 'bg-violet-brand text-white'
                  : step === ProgressStepStatus.Successful
                  ? 'bg-green-check text-white'
                  : 'bg-gray-light text-black'
              }`}
            />
          )}
        </>
      ))}
    </div>
  );
};

export default ProgressIndicator;
