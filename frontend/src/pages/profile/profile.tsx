import { observer } from 'mobx-react-lite';
import { DataProfile } from '#/components';
import { ButtonSignout } from '#/components/buttonSignout';
import './styles.css';
import Navbar  from '#/components/navbar/navbar';
const ProfilePage = () => {
  const logged: boolean = true;
  return (
    <div>
      {logged && <Navbar />}
      <main className="min__height-main flex justify-between flex-col px-4 profile__design">
        <DataProfile />
        <ButtonSignout />
      </main>
    </div>
  );
};

export const Profile = observer(ProfilePage);

export default Profile;
