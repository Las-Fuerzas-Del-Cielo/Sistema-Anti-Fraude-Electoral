import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './Exports';
interface Props {
  provincias: string[];
}

function Selector(props: Props) {
  return (
    <Select>
      <SelectTrigger className='w-[180px]'>
        <SelectValue placeholder='Filtrar por Provincia' />
      </SelectTrigger>
      <SelectContent>
        {props.provincias.map((provincia) => (
          <SelectItem key={self.crypto.randomUUID()} value={provincia}>
            {provincia}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export default Selector;
