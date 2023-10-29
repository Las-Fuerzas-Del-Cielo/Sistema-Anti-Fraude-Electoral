import { FC } from 'react';
import { Link } from 'react-router-dom';
import { observer } from 'mobx-react';
import Button from '#/components/button';
import FormHeader from '#/components/formHeader';
import FlatList from '#/components/flatList';
import ProgressIndicator from '#/components/progressIndicator';

import { FlatListTypeEnum } from '#/components/flatList/types';
import { ProgressStepStatus } from '#/components/progressIndicator/types';
import { ILoadDataProps } from './types';

import './styles.css';

const LoadDataPage: FC<ILoadDataProps> = ({ message }) => {
  return (
    <section className="bg-white items-center flex flex-col ">
      <FormHeader routerLink="/" title="Cargar datos del certificado" />
      <div className="container mx-auto p-2">
        <div className="flex items-center justify-center my-210">
          <ProgressIndicator
            steps={[
              ProgressStepStatus.Successful,
              ProgressStepStatus.Successful,
              ProgressStepStatus.Active,
            ]}
          />
        </div>
        <div className="flex items-center justify-center my-20 w-full">
          <FlatList
            logo="src/assets/logos/uxp.svg"
            type={FlatListTypeEnum.massa}
            subTitle="Massa"
            title="Sergio Tomas"
            votes={0}
            edit={true}
          />
        </div>
        <div className="flex items-center justify-center my-20 w-full">
          <FlatList
            logo="src/assets/logos/lla-logo.svg"
            type={FlatListTypeEnum.milei}
            subTitle="Milei"
            title="Javier Gerardo"
            votes={0}
            edit={true}
          />
        </div>
        <div className="flex items-center justify-center my-20">
          {/* TODO: Mover a Dashboard */}
          <Link to="/">
            <Button
              className="bg-violet-brand p-4 text-white rounded-xl font-semibold text-xl tracking-wider w-full"
              type="submit"
              label="Enviar Datos"
            />
          </Link>
        </div>
      </div>
    </section>
  );
};

export const LoadData = observer(LoadDataPage);

export default LoadData;
