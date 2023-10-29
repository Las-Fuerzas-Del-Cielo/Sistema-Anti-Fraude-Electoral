import { DataProfile } from '#/components';
import { ButtonSignout } from '#/components/buttonSignout';
import { observer } from 'mobx-react-lite';

const ProfilePage = () => {
  return (
    <main
      style={{ backgroundColor: '#F1F3F4', paddingTop: '1.5em' }}
      className='minHeight-main flex justify-between flex-col px-4'
    >
      <DataProfile />
      <ButtonSignout />
    </main>
  );
};

export const Profile = observer(ProfilePage);

export default Profile;
