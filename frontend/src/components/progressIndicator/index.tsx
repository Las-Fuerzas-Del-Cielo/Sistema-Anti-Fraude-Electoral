import './styles.css'

export const ProgressIndicator = ({ stateStep }: any) => {
  return (
    <div className="w-full flex items-center">
      {/* Step 1 */}
      <div
        className={`circle flex justify-center items-center rounded-full ${
          stateStep.step_one === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_one === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_one === 'successful' ? (
          <img
            className="icon-tick"
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '1'
        )}
      </div>

      <div
        className={`tick ${
          stateStep.step_one === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_one === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      ></div>

      <div
        className={`tick ${
          stateStep.step_two === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_two === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      ></div>

      {/* Step 2 */}
      <div
        className={`circle flex justify-center items-center rounded-full ${
          stateStep.step_two === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_two === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_two === 'successful' ? (
          <img
            className="icon-tick"
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '2'
        )}
      </div>

      <div
        className={`tick ${
          stateStep.step_two === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_two === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      ></div>

      <div
        className={`tick ${
          stateStep.step_three === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_three === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      ></div>

      {/* Step 3 */}
      <div
        className={`circle flex justify-center items-center rounded-full ${
          stateStep.step_three === 'active'
            ? 'bg-violet text-white'
            : stateStep.step_three === 'successful'
            ? 'bg-green text-white'
            : 'bg-light-gray text-black'
        }`}
      >
        {stateStep.step_three === 'successful' ? (
          <img
            className="icon-tick"
            src="src/assets/check-icono.svg"
            alt=""
          />
        ) : (
          '3'
        )}
      </div>
    </div>
  );
};
