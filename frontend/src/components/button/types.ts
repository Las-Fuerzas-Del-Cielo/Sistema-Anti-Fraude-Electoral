export interface IButtonProps {
  type: 'button' | 'submit' | 'reset';
  className: string;
  label: string;
  // eslint-disable-next-line no-unused-vars
  onClick?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}
