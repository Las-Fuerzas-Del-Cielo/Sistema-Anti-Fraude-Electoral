import { useState } from "react";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from "./Exports"
  interface Props{
    placeholder: string,
    provincias: string[]
  }

  function Selector(props:Props) {
    // TODO: HANDLE VALUE FROM PARENT BY PROPS
    const [value, setValue] = useState<string>('');
    
    return (
        <Select onValueChange={aux => setValue(aux)} >
        <SelectTrigger id={'selector'}>
        <label
        htmlFor={'selector'}
        className={`absolute left-0 px-3 transition-all duration-300 ease-in-out transform ${
          value
            ? "-translate-y-1/3 text-xs font-thin"
            : " text-md"
        } pointer-events-none`}
      >
        {props.placeholder}
      </label>
          <span  className="transform translate-y-1/2">
            {value}
          </span>
        </SelectTrigger>
        <SelectContent>
            {props?.provincias?.map((provincia, index) => (
                <SelectItem value={provincia} key={`${provincia} ${index}`} >{provincia}</SelectItem>
            ))}
        </SelectContent>
      </Select>      
    )
  }
  
  export default Selector
  