import { IUser } from '#/interfaces/IUser';

export interface IProfileDataProps {
  user: IUser;
}

export interface IProfileDataTableProps {
  title: string;
  text: string;
}

export interface IFieldProps {
  fieldText: IProfileDataTableProps;
  isLast: boolean;
}
