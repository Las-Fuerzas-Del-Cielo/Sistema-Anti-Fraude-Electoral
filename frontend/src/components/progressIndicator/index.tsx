export const ProgressIndicator = ({ stateStep }: any) => {
  return (
    <div className="w-full flex items-center">
      {/* Step 1 */}
      <div
        style={{ width: "32px", height: "32px" }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_one === "active"
            ? "bg-violet text-white"
            : stateStep.step_one === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
      >
        1
      </div>

      <div
        className={`${
          stateStep.step_one === "active"
            ? "bg-violet text-white"
            : stateStep.step_one === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
        style={{ content: "", width: "25px", height: "2px" }}
      ></div>
      
      <div
        className={`${
          stateStep.step_two === "active"
            ? "bg-violet text-white"
            : stateStep.step_two === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
        style={{ content: "", width: "25px", height: "2px" }}
      ></div>

      {/* Step 2 */}
      <div
        style={{ width: "32px", height: "32px" }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_two === "active"
            ? "bg-violet text-white"
            : stateStep.step_two === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
      >
        2
      </div>

      <div
        className={`${
          stateStep.step_two === "active"
            ? "bg-violet text-white"
            : stateStep.step_two === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
        style={{ content: "", width: "25px", height: "2px" }}
      ></div>
      
      <div
        className={`${
          stateStep.step_three === "active"
            ? "bg-violet text-white"
            : stateStep.step_three === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
        style={{ content: "", width: "25px", height: "2px" }}
      ></div>

      {/* Step 3 */}
      <div
        style={{ width: "32px", height: "32px" }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_three === "active"
            ? "bg-violet text-white"
            : stateStep.step_three === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
      >
        3
      </div>

      <div
        className={`${
          stateStep.step_three === "active"
            ? "bg-violet text-white"
            : stateStep.step_three === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
        style={{ content: "", width: "25px", height: "2px" }}
      ></div>
      
      <div
        className={`${
          stateStep.step_three === "active"
            ? "bg-violet text-white"
            : stateStep.step_three === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
        style={{ content: "", width: "25px", height: "2px" }}
      ></div>

      {/* Step 4 */}
      <div
        style={{ width: "32px", height: "32px" }}
        className={`flex justify-center items-center rounded-full ${
          stateStep.step_four === "active"
            ? "bg-violet text-white"
            : stateStep.step_four === "successful"
            ? "bg-green text-white"
            : "bg-light-gray text-black"
        }`}
      >
        4
      </div>
    </div>
  );
};
