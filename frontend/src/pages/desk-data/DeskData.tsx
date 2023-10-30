import Button from '#/components/button';
import FormHeader from '#/components/formHeader';
import ProgressIndicator from '#/components/progressIndicator';
import { ProgressStepStatus } from '#/components/progressIndicator/types';
import { Selector } from '#/components/selector';
import { observer } from 'mobx-react-lite';

const DeskData = () => {
  const DummyData = ['Example', 'Example 1', 'Example 2'];

  return (
    <>
      <main className='items-center flex flex-col relative'>
        <FormHeader routerLink='/' title='' />

        <div className='md:w-1/2 w-full rounded-xl z-10 mt-10 px-10 my-10'>
          <ProgressIndicator
            steps={[
              ProgressStepStatus.Active,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
            ]}
          />
        </div>

        <section className='md:w-1/2 w-full rounded-xl z-10 mt-10 px-10 my-10'>
          <h1 className='text-xl font-bold mb-3 my-2'>Ubicacion de la mesa</h1>
          <h2 className='p-2 text-start text-sm-lg'>
            Recopilación de la <b>ubicación</b> precisa del <b>centro educativo.</b>
          </h2>

          <div className=''>
            <Selector placeholder='Distrito' provincias={DummyData} />
            <Selector placeholder='Sección Electoral' provincias={DummyData} />
            <Selector placeholder='Sección' provincias={DummyData} />
            <Selector placeholder='Municipio' provincias={DummyData} />
            <Selector placeholder='Establecimiento' provincias={DummyData} />
          </div>

          <Button
            className='mt-10 bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider'
            label='Continuar'
            type='submit'
          />
        </section>
      </main>
    </>
  );
};

export const deskData = observer(DeskData);

export default DeskData;
