enum ProgressIndicatorStatus {
    Active = 'active',
    Successful = 'successful'
  }
  
export interface IProgressIndicatorProps {
    step_one: ProgressIndicatorStatus;
    step_two?: ProgressIndicatorStatus;
    step_three?: ProgressIndicatorStatus;
    step_four?: ProgressIndicatorStatus;
  }