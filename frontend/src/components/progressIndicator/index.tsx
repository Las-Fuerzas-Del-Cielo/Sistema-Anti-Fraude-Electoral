export const ProgressIndicator = ({ stateStep }: any) => {
  return (
    <div className="w-full flex items-center" style={{ padding: '16px' }}>
      {/* Step 1 */}
      <div
        style={{ width: '32px', height: '32px' }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_one === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_one === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_one === 'successful' ? (
          <img
            style={{ width: '50%', height: '50%' }}
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '1'
        )}
      </div>

      <div
        className={`${
          stateStep.step_one === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_one === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
        style={{ content: '', width: '25px', height: '2px' }}
      ></div>

      <div
        className={`${
          stateStep.step_two === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_two === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
        style={{ content: '', width: '25px', height: '2px' }}
      ></div>

      {/* Step 2 */}
      <div
        style={{ width: '32px', height: '32px' }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_two === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_two === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_two === 'successful' ? (
          <img
            style={{ width: '50%', height: '50%' }}
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '2'
        )}
      </div>

      <div
        className={`${
          stateStep.step_two === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_two === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
        style={{ content: '', width: '25px', height: '2px' }}
      ></div>

      <div
        className={`${
          stateStep.step_three === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_three === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
        style={{ content: '', width: '25px', height: '2px' }}
      ></div>

      {/* Step 3 */}
      <div
        style={{ width: '32px', height: '32px' }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_three === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_three === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_three === 'successful' ? (
          <img
            style={{ width: '50%', height: '50%' }}
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '3'
        )}
      </div>

      <div
        className={`${
          stateStep.step_three === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_three === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
        style={{ content: '', width: '25px', height: '2px' }}
      ></div>

      <div
        className={`${
          stateStep.step_four === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_four === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
        style={{ content: '', width: '25px', height: '2px' }}
      ></div>

      {/* Step 4 */}
      <div
        style={{ width: '32px', height: '32px' }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_four === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_four === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_four === 'successful' ? (
          <img
            style={{ width: '50%', height: '50%' }}
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '4'
        )}
      </div>
    </div>
  );
};
