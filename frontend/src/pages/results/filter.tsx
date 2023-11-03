import { useState } from 'react';
import { observer } from 'mobx-react-lite';
import { Selector } from '#/components/selector';
import Button from '#/components/button';
import Navbar from '#/components/navbar';
import { useSelectData } from '#/hooks/utils/useSelectData';

const FilterPage = () => {
  const [distrito, setDistrito] = useState<string>('');
  const [seccionElectoral, setSeccionElectoral] = useState<string>('');
  const [seccion, setSeccion] = useState<string>('');
  const [municipio, setMunicipio] = useState<string>('');
  const [circuito, setCircuito] = useState<string>('');
  const [establecimiento, setEstablecimiento] = useState<string>('');
  const [mesa, setMesa] = useState<string>('');

  const {
    districts,
    electoralSections,
    sections,
    municipalities,
    establishments,
    circuits,
    tables,
  } = useSelectData();

  return (
    <>
      <Navbar />
      <main className="items-center flex flex-col relative px-10">
        <section className="md:w-1/2 w-full rounded-xl z-10 mt-10">
          <h1 className="text-xl font-bold mb-6">Resultados totales</h1>
          <div className="px-3">
            <Selector
              options={districts}
              placeholder="Distrito"
              value={distrito}
              onChange={setDistrito}
            />
            <Selector
              options={electoralSections}
              placeholder="Sección Electoral"
              value={seccionElectoral}
              onChange={setSeccionElectoral}
            />
            <Selector
              options={sections}
              placeholder="Sección"
              value={seccion}
              onChange={setSeccion}
            />
            <Selector
              options={municipalities}
              placeholder="Municipio"
              value={municipio}
              onChange={setMunicipio}
            />
            <Selector
              options={circuits}
              placeholder="Circuito "
              value={circuito}
              onChange={setCircuito}
            />
            <Selector
              options={establishments}
              placeholder="Establecimiento!!!"
              value={establecimiento}
              onChange={setEstablecimiento}
            />
            <Selector
              options={tables}
              placeholder="Mesa"
              value={mesa}
              onChange={setMesa}
            />
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
