import { Selector } from "#/components/selector";
import Button from "#/components/button";
import { observer } from "mobx-react-lite";
import Navbar from "#/components/navbar";

const FilterPage = () => {
  const DummyData = ["Example", "Example 1", "Example 2"];

  return (
    <>
    <Navbar />
    <main className="items-center flex flex-col relative px-10">
      <section className="md:w-1/2 w-full rounded-xl z-10 mt-10">
        <h1 className="text-xl font-bold mb-6">Resultados totales</h1>
        <div className="px-3">
          <Selector provincias={DummyData} placeholder="Distrito" />
          <Selector provincias={DummyData} placeholder="Sección Electoral" />
          <Selector provincias={DummyData} placeholder="Sección" />
          <Selector provincias={DummyData} placeholder="Municipio" />
          <Selector provincias={DummyData} placeholder="Circuito " />
          <Selector provincias={DummyData} placeholder="Establecimiento" />
          <Selector provincias={DummyData} placeholder="Mesa" />
        </div>

        <Button
          className="mt-10 bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider"
          type="submit"
          label="Aplicar Filtros"
          />
        <Button
            className="border-2 border-red text-red bg-transparent p-3 w-full rounded-xl text-xl tracking-wider shadow-md hover:border-violet-light mt-3"
            type="submit"
            label="Alertar Irregularidades"
          />
      </section>
    </main>
    </>
  );
};

export const Filter = observer(FilterPage);

export default Filter;
