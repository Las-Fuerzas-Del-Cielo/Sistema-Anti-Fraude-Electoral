export interface IButtonProps {
  type: 'button' | 'submit' | 'reset';
  className: string;
  label: string;
  // eslint-disable-next-line no-unused-vars
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}
