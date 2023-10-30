export enum ProgressStepStatus {
  Pending = 'pending',
  Active = 'active',
  Successful = 'successful',
}

export interface IProgressIndicatorProps {
  steps: ProgressStepStatus[];
}
