import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import './custom.css';
import { StoreProvider } from './store/index.js';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StoreProvider>
      <App />
    </StoreProvider>
  </React.StrictMode>
);
