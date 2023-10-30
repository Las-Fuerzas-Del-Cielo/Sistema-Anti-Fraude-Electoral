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
    <main className="items-center flex flex-col relative">
      <FormHeader routerLink="/" title="" />

      <div className= "md:w-1/2 w-full rounded-xl z-10 mt-10 px-10 my-10">

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
          Recopilación de la <b>ubicación</b> precisa del <b>centro educativo.</b>
        </h2>

        <div className="">
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
