export interface IInputProps {
  label: string;
  type: 'text' | 'password';
  id: string;
  placeholder: string;
  error?: boolean;

  className?: string;
  labelClassName?: string;
  inputClassName?: string;

  appearance?: 'outline' | 'underline';
  // eslint-disable-next-line no-unused-vars
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}
