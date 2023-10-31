import { IUser } from '#/interfaces/IUser';
import {
  IFieldProps,
  IProfileDataProps,
  IProfileDataTableProps,
} from './types';

export const DataProfile = ({ user }: IProfileDataProps) => {
  const profileData: IProfileDataTableProps[] = [
    { title: 'Nombres', text: user.firstName },
    { title: 'Apellido', text: user.lastName },
    { title: 'Email', text: user.email },
    { title: 'DNI', text: user.dni },
    { title: 'Provincia', text: user.province },
    { title: 'Circuito', text: user.circuit },
    { title: 'Mesa', text: user.table },
  ];

  return (
    <section className="flex flex-col w-full rounded-lg px-4 py-2 gap-x-4 bg-gray-100">
      {profileData?.map((fieldText, index) => (
        <FieldText
          fieldText={fieldText}
          isLast={index === profileData.length - 1}
          key={crypto.randomUUID()}
        />
      ))}
    </section>
  );
};

export const FieldText = ({ fieldText, isLast }: IFieldProps) => {
  return (
    <article
      className={`flex w-full justify-between ${
        isLast ? 'border-b-0' : 'border-b border-gray-300'
      }`}
    >
      <div className="w-2/4 flex justify-start py-4">
        <span className="font-semibold">{fieldText.title}</span>
      </div>

      <div className="w-2/4 flex justify-end py-4">
        <span className="font-normal">{fieldText.text}</span>
      </div>
    </article>
  );
};
