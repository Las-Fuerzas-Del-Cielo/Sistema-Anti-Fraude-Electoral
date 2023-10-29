export interface IInputProps {
  label: string;
  type: 'text' | 'password';
  id: string;
  placeholder: string;

  className?: string;
  labelClassName?: string;
  inputClassName?: string;

  appearance?: 'outline' | 'underline'
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}
