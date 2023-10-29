import { DataProfile } from '#/components';
import { ButtonSignout } from '#/components/buttonSignout';

export const Perfil = () => {
  return (
    <main style={{ backgroundColor: '#F1F3F4', paddingTop: '21px' }} className="minHeight-main flex justify-between flex-col px-4">
      <DataProfile />
      <ButtonSignout />
    </main>
  );
};
