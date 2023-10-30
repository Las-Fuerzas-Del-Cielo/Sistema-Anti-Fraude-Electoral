import { useCallback, useEffect, useState } from 'react';
import { useEventListener } from './use-event-listener';

interface Size {
  width: number;
  height: number;
}

// eslint-disable-next-line no-unused-vars
export function useElementSize<T extends HTMLElement = HTMLDivElement>(): [(node: T | null) => void, Size] {
  const [ref, setRef] = useState<T | null>(null);
  const [size, setSize] = useState<Size>({
    width: 0,
    height: 0,
  });

  const handleSize = useCallback(() => {
    setSize({
      width: ref?.offsetWidth || 0,
      height: ref?.offsetHeight || 0,
    });
  }, [ref?.offsetHeight, ref?.offsetWidth]);

  useEventListener('resize', handleSize);

  useEffect(() => {
    handleSize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref?.offsetHeight, ref?.offsetWidth]);

  return [setRef, size];
}

// usage :  const [ref, { width, height }] = useElementSize()
// Le pasamos la ref del elemento que queremos el tamaño y nos devuelve el width y height
