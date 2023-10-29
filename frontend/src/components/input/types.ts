export interface IInputProps {
  type: 'text' | 'password';
  id: string;
  placeholder: string;
  className: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}
