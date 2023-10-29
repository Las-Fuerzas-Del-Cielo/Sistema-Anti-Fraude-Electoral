/* eslint-disable react-refresh/only-export-components */
import { createContext, FC, ReactNode, useContext } from 'react';
import { RootStore, store } from './root';

const StoreContext = createContext<RootStore | undefined>(undefined);

interface Properties {
  children: ReactNode;
}

export const StoreProvider: FC<Properties> = ({ children }) => {
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
};

export const useStore = () => {
  const context = useContext(StoreContext);

  if (context === undefined) {
    throw new Error('useStore must be used within StoreProvider');
  }

  return context;
};

export * from './root';
