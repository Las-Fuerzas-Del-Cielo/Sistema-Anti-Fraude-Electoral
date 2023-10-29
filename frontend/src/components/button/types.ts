export interface IButtonProps {
  type: 'button' | 'submit' | 'reset';
  className: string;
  label: string;
  onClick?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}
