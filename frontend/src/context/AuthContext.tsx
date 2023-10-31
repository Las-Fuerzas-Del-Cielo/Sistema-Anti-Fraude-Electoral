import {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from 'react';
import useAxios from '#/hooks/utils/useAxios';
import { IUser } from '#/interfaces/IUser';
import { useNavigate } from 'react-router-dom';

interface AuthContextType {
  isAuthenticated: boolean;
  user: IUser | null;
  login: (user: IUser) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const axios = useAxios();
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<IUser | null>(null);

  useEffect(() => {
    refreshToken();
  }, []);

  const login = (userData: IUser) => {
    sessionStorage.setItem('accessToken', userData.jwt);
    setIsAuthenticated(true);
    setUser(userData);
  };

  const logout = () => {
    sessionStorage.removeItem('accessToken');
    setIsAuthenticated(false);
    setUser(null);
    navigate('/login');
  };

  const refreshToken = async () => {
    const accessToken = sessionStorage.getItem('accessToken');
    if (!accessToken) return logout();

    const { data, error } = await axios.get('/refreshToken');
    if (error) return logout();

    login(data);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth debe ser usado dentro de un AuthProvider');
  }
  return context;
};
