import { Select, SelectContent, SelectItem, SelectTrigger } from './Exports';

interface ISelectorProps {
  placeholder: string;
  options: { key: string; label: string }[];
  value?: string;
  onChange?: (value: string) => void;
}

function Selector({ placeholder, options, value, onChange }: ISelectorProps) {
  return (
    <Select onValueChange={onChange}>
      <SelectTrigger id={'selector'}>
        <label
          htmlFor={'selector'}
          className={`absolute left-0 px-3 transition-all duration-300 ease-in-out transform ${
            value ? '-translate-y-1/3 text-xs font-thin' : ' text-md'
          } pointer-events-none`}
        >
          {placeholder}
        </label>
        <span className="transform translate-y-1/2">{value}</span>
      </SelectTrigger>
      <SelectContent>
        {options?.map((option, index) => (
          <SelectItem value={option.key} key={`${option.key} ${index}`}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export default Selector;
