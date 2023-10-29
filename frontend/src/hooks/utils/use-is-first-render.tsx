import { useRef } from "react";

export const useIsFirstRender = (): boolean => {
  const isFirst = useRef(true);

  if (isFirst.current) {
    isFirst.current = false;

    return true;
  }

  return isFirst.current;
};

// usage :  const isFirst = useIsFirstRender()
