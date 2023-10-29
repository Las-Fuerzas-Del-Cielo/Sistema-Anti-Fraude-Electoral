import { Dispatch, SetStateAction, useCallback, useState } from "react";

export const useToggle = (
  defaultValue?: boolean
): [boolean, () => void, Dispatch<SetStateAction<boolean>>] => {
  const [value, setValue] = useState(defaultValue ?? false);

  const toggle = useCallback(() => setValue((prevValue) => !prevValue), []);

  return [value, toggle, setValue];
};
