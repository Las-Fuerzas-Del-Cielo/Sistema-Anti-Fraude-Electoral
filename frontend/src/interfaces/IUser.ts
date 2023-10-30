export interface IUser {
  id: number;
  firstName: string;
  lastName: string;
  jwt: string;
  email: string;
  password: string;
  dni: string;
  phone: string;
  address: string;
  role: string;

  province: string;
  circuit: string;
  table: string;

  createdAt: string;
  updatedAt: string;
}
