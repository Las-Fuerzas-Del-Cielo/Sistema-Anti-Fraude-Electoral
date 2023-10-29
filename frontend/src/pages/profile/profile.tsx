import { observer } from 'mobx-react-lite';
import { DataProfile } from '#/components';
import { ButtonSignout } from '#/components/buttonSignout';
import './styles.css';

const ProfilePage = () => {
  return (
    <main className='min__height-main flex justify-between flex-col px-4 profile__design'>
      <DataProfile />
      <ButtonSignout />
    </main>
  );
};

export const Profile = observer(ProfilePage);

export default Profile;
