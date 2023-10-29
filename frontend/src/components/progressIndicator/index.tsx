import React from 'react';
import { IProgressIndicatorProps, ProgressStepStatus } from './types';
import './styles.css';

const ProgressIndicator = ({ steps }: IProgressIndicatorProps) => {
  return (
    <div className="w-full flex justify-between items-center px-20 mt-4">
      {steps.map((step, index) => (
        <React.Fragment key={index}>
          <div
            className={`circle flex justify-center items-center rounded-full ${
              step === ProgressStepStatus.Active
                ? 'bg-violet-brand text-white'
                : step === ProgressStepStatus.Successful
                ? 'bg-green-check text-white'
                : 'bg-gray-light text-black'
            }`}
          >
            {step === ProgressStepStatus.Successful ? (
              <img
                className="w-4 h-4"
                src="src/assets/check-icono.svg"
                alt=""
              />
            ) : (
              <span className="font-normal text-xl">
                {(index + 1).toString()}
              </span>
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
            ></div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export default ProgressIndicator;
