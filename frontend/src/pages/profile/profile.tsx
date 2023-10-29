import { DataProfile } from '#/components';
import { ButtonSignout } from '#/components/buttonSignout';
import Header from '#/components/formHeader';
import { observer } from 'mobx-react-lite';

import './styles.css';

const ProfilePage = () => {
  return (
    <>
      <Header routerLink='/dashboard' title='Mi Perfil' />
      <main className='min__height-main flex justify-between flex-col px-4 profile__design bg-white'>
        <DataProfile />
        <ButtonSignout />
      </main>
    </>
  );
};

export const Profile = observer(ProfilePage);

export default Profile;
