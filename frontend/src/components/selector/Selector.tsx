import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
  } from "./Exports"
  interface Props{
    provincias: string[]
  }

  function Selector(props:Props) {
    return (
        <Select>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Filtrar por Provincia" />
        </SelectTrigger>
        <SelectContent>
            {props.provincias.map((provincia) => (
                <SelectItem value={provincia} key={self.crypto.randomUUID()} >{provincia}</SelectItem>
            ))}
        </SelectContent>
      </Select>      
    )
  }
  
  export default Selector
  