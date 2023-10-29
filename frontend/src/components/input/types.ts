export interface IInputProps {
  label: string;
  type: 'text' | 'password';
  id: string;
  placeholder: string;
  className: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}
