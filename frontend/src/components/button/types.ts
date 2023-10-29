export interface ButtonProps {
  label: string;
  onClick: () => void;
  className?: string;
  variant?: 'default' | 'secondary' | 'inactive' | 'warning' | 'outline';
  type?: 'button' | 'submit' | 'reset';
}
