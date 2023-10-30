import { Selector } from "#/components/selector";
import Button from "#/components/button";
import { observer } from "mobx-react-lite";
import Navbar from "#/components/navbar";
import ProgressIndicator from '#/components/progressIndicator';
import FormHeader from '#/components/formHeader';
import { ProgressStepStatus } from '#/components/progressIndicator/types';


const DeskData = () => {
  const DummyData = ["Example", "Example 1", "Example 2"];

  return (
    <>
    <FormHeader routerLink="/" title="" />
    <main className="items-center flex flex-col relative px-10">
      <section className="md:w-1/2 w-full rounded-xl z-10 mt-10">
        
      <ProgressIndicator
            steps={[
              ProgressStepStatus.Active,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
            ]}
          />
        <h1 className="text-xl font-bold mb-6 my-10">Ubicacion de la mesa</h1>
        <h2 className="p-4 mx-10 text-start mx-12 text-xl">Recopilación de la ubicación precisa del centro educativo</h2>
        
        <div className="px-3">
          <Selector provincias={DummyData} placeholder="Distrito" />
          <Selector provincias={DummyData} placeholder="Sección Electoral" />
          <Selector provincias={DummyData} placeholder="Sección" />
          <Selector provincias={DummyData} placeholder="Municipio" />
          <Selector provincias={DummyData} placeholder="Establecimiento" />
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
