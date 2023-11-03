import { Selector } from '#/components/selector';
import Button from '#/components/button';
import { observer } from 'mobx-react-lite';
import Navbar from '#/components/navbar';
import ProgressIndicator from '#/components/progressIndicator';
import FormHeader from '#/components/formHeader';
import { ProgressStepStatus } from '#/components/progressIndicator/types';
import { useSelectData } from '../../hooks/utils/useSelectData';

const DeskData = () => {
  const {
    districts,
    electoralSections,
    sections,
    municipalities,
    establishments,
  } = useSelectData();

  return (
    <>
      <main className="items-center flex flex-col relative">
        <FormHeader routerLink="/" title="" />

        <div className="md:w-1/2 w-full rounded-xl z-10 mt-10 px-10 my-10">
          <ProgressIndicator
            steps={[
              ProgressStepStatus.Active,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
            ]}
          />
        </div>

        <section className="md:w-1/2 w-full rounded-xl z-10 mt-10 px-10 my-10">
          <h1 className="text-xl font-bold mb-3 my-2">Ubicacion de la mesa</h1>
          <h2 className="p-2 text-start text-sm-lg">
            Recopilación de la <b>ubicación</b> precisa del{' '}
            <b>centro educativo.</b>
          </h2>

          <div className="">
            <Selector options={districts} placeholder="Distrito" />
            <Selector options={electoralSections} placeholder="Sección Electoral" />
            <Selector options={sections} placeholder="Sección" />
            <Selector options={municipalities} placeholder="Municipio" />
            <Selector options={establishments} placeholder="Establecimiento" />
          </div>

          <Button
            className="mt-10 bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider"
            type="submit"
            label="Continuar"
          />
        </section>
      </main>
    </>
  );
};

export const deskData = observer(DeskData);

export default DeskData;
