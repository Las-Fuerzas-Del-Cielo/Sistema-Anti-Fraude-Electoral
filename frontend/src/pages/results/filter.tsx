import { Selector } from '#/components/selector';
import Button from '#/components/button';
import { observer } from 'mobx-react-lite';
import Navbar from '#/components/navbar';
import { useState } from 'react';

const dummyData = [
  { key: 'ex1', label: 'Example' },
  { key: 'ex2', label: 'Example 2' },
  { key: 'ex3', label: 'Example 3' },
];

const FilterPage = () => {
  const [distrito, setDistrito] = useState('');
  const [seccionElectoral, setSeccionElectoral] = useState('');
  const [seccion, setSeccion] = useState('');
  const [municipio, setMunicipio] = useState('');
  const [circuito, setCircuito] = useState('');
  const [establecimiento, setEstablecimiento] = useState('');
  const [mesa, setMesa] = useState('');

  return (
    <>
      <Navbar />
      <main className="items-center flex flex-col relative px-10">
        <section className="md:w-1/2 w-full rounded-xl z-10 mt-10">
          <h1 className="text-xl font-bold mb-6">Resultados totales</h1>
          <div className="px-3">
            <Selector
              options={dummyData}
              placeholder="Distrito"
              value={distrito}
              onChange={setDistrito}
            />
            <Selector
              options={dummyData}
              placeholder="Sección Electoral"
              value={seccionElectoral}
              onChange={setSeccionElectoral}
            />
            <Selector
              options={dummyData}
              placeholder="Sección"
              value={seccion}
              onChange={setSeccion}
            />
            <Selector
              options={dummyData}
              placeholder="Municipio"
              value={municipio}
              onChange={setMunicipio}
            />
            <Selector
              options={dummyData}
              placeholder="Circuito "
              value={circuito}
              onChange={setCircuito}
            />
            <Selector
              options={dummyData}
              placeholder="Establecimiento"
              value={establecimiento}
              onChange={setEstablecimiento}
            />
            <Selector
              options={dummyData}
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
