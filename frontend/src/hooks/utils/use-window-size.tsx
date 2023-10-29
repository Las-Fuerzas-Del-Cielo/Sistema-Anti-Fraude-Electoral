import { useState } from 'react';
import { useEffectOnce } from './use-effect-once';
import { useEventListener } from './use-event-listener';

interface WindowSize {
  width: number;
  height: number;
}

export function useWindowSize(): WindowSize {
  const [windowSize, setWindowSize] = useState<WindowSize>({
    width: 0,
    height: 0,
  });

  const handleSize = () => {
    setWindowSize({
      width: window.innerWidth,
      height: window.innerHeight,
    });
  };

  useEventListener('resize', handleSize);

  useEffectOnce(() => {
    handleSize();
  });

  return windowSize;
}
