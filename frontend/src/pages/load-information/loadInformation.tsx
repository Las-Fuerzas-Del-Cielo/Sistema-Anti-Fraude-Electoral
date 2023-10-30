import { FC } from 'react';
import { Link } from 'react-router-dom';
import { observer } from 'mobx-react';
import Button from '#/components/button';
import FormHeader from '#/components/formHeader';
import FlatList from '#/components/flatList';
import ProgressIndicator from '#/components/progressIndicator';

import { FlatListTypeEnum } from '#/components/flatList/types';
import { ProgressStepStatus } from '#/components/progressIndicator/types';
import { ILoadInformationProps } from './types';

import { useState } from 'react';

import './styles.css';

const LoadInformationPage: FC<ILoadInformationProps> = ({ message }) => {
  // Migrate to global state later?
  const [circuit, setCircuit] = useState<number>(0);
  const [table, setTable] = useState<number>(0);
  const [electors, setElectors] = useState<number>(0);
  const [envelopes, setEnvelopes] = useState<number>(0);

  const [totalVotes, setTotalVotes] = useState<number>(0);
  const [correctData, setCorrectData] = useState<boolean>(false);

  const handleCircuitChange = (value: number) => {
    const newValue: number = value;
    if (newValue >= 0) {
      setCircuit(newValue);
    }
  };

  const handleTableChange = (value: number) => {
    const newValue: number = value;
    if (newValue >= 0) {
      setTable(newValue);
    }
  };

  const handleElectorsChange = (value: number) => {
    const newValue: number = value;
    if (newValue >= 0) {
      setElectors(newValue);
    }
  };

  const handleEnvelopesChange = (value: number) => {
    const newValue: number = value;
    if (newValue >= 0) {
      setEnvelopes(newValue);
    }
  };

  const updateTotalVotes = (newValue: number) => {
    setTotalVotes((prevTotal: number) => prevTotal + newValue);
  };

  const handleCheckbox = () => {
    setCorrectData((correctData) => !correctData);
  };

  // Conditional styles
  const selectedInputStyle: string = 'border-2 border-violet-brand !text-black';

  const circuitInputStyle: string | null =
    circuit > 0 ? selectedInputStyle : null;
  const tableInputStyle: string | null = table > 0 ? selectedInputStyle : null;
  const electorsInputStyle: string | null =
    electors > 0 ? selectedInputStyle : null;
  const envelopesInputStyle: string | null =
    envelopes > 0 ? selectedInputStyle : null;

  const electorsEnvelopesDiffStyle: string | null =
    electors - envelopes > 4 ? 'text-red' : null;

  const totalVotesDiffStyle: string | null =
    envelopes - totalVotes != 0 ? '!text-red' : null;

  const flatList = [
    {
      logo: 'src/assets/logos/lla-logo.svg',
      type: FlatListTypeEnum.milei,
      subTitle: 'Milei',
      title: 'Javier Gerardo',
      votes: 0,
      edit: true,
    },
    {
      logo: 'src/assets/logos/uxp.svg',
      type: FlatListTypeEnum.massa,
      subTitle: 'Massa',
      title: 'Sergio Tomas',
      votes: 0,
      edit: true,
    },
    {
      logo: 'src/assets/icon/mail-invalid.svg',
      type: FlatListTypeEnum.null,
      subTitle: '',
      title: 'Votos nulos',
      votes: 0,
      edit: true,
    },
    {
      logo: 'src/assets/icon/mail-appealed.svg',
      type: FlatListTypeEnum.appealed,
      subTitle: '',
      title: 'Votos recurridos',
      votes: 0,
      edit: true,
    },
    {
      logo: 'src/assets/icon/mail-contested.svg',
      type: FlatListTypeEnum.contested,
      subTitle: '',
      title: 'Votos identidad impugnada',
      votes: 0,
      edit: true,
    },
    {
      logo: 'src/assets/icon/mail-person.svg',
      type: FlatListTypeEnum.electoralCommand,
      subTitle: '',
      title: 'Votos de comando electoral',
      votes: 0,
      edit: true,
    },
    {
      logo: 'src/assets/icon/mail-closed.svg',
      type: FlatListTypeEnum.blank,
      subTitle: '',
      title: 'Votos en blanco',
      votes: 0,
      edit: true,
    },
  ];

  return (
    <section className="bg-white items-center flex flex-col">
      <FormHeader routerLink="/" title="" />
      <div className="container mx-auto p-2">
        <div className="flex items-center justify-center my-210">
          <ProgressIndicator
            steps={[
              ProgressStepStatus.Successful,
              ProgressStepStatus.Successful,
              ProgressStepStatus.Successful,
              ProgressStepStatus.Active,
            ]}
          />
        </div>
        <div className="py-8 text-neutral-700 text-3xl font-bold">
          Cargar datos del certificado
        </div>
        <div className="flex flex-row w-full justify-center gap-16">
          <div>
            <div className="text-violet-brand font-bold text-xl my-2">
              Circuito
            </div>
            <input
              type="number"
              value={circuit}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleCircuitChange(Number(e.target.value))
              }
              className={`border-2 text-center border-gray-300 outline-none cursor-default bg-white text-neutral-500 font-bold rounded-xl h-12 w-32 flex text-2xl ${circuitInputStyle}`}
            />
          </div>
          <div>
            <div className="text-violet-brand font-bold text-xl my-2">Mesa</div>
            <input
              type="number"
              value={table}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleTableChange(Number(e.target.value))
              }
              className={`border-2 text-center border-gray-300 outline-none cursor-default bg-white text-neutral-500 font-bold rounded-xl h-12 w-32 flex text-2xl ${tableInputStyle}`}
            />
          </div>
        </div>
        <div className="flex items-center justify-center w-full p-2">
          <div className="flex p-2 justify-between items-center w-full  max-w-md ">
            <div className="text-2xl text-neutral-700 font-bold px-3 py-5 tracking-wide">
              Cantidad de electores
            </div>
            <input
              type="number"
              value={electors}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleElectorsChange(Number(e.target.value))
              }
              className={`border-2 text-center border-gray-300 outline-none cursor-default bg-white text-neutral-500 font-bold rounded-xl h-12 w-32 flex text-2xl ${electorsInputStyle}`}
            />
          </div>
        </div>
        <div className="flex items-center justify-center w-full p-2">
          <div className="flex p-2 justify-between items-center w-full  max-w-md ">
            <div className="text-2xl text-neutral-700 font-bold px-3 py-5 tracking-wide">
              Cantidad de sobres
            </div>
            <input
              type="number"
              value={envelopes}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleEnvelopesChange(Number(e.target.value))
              }
              className={`border-2 text-center border-gray-300 outline-none cursor-default bg-white text-neutral-500 font-bold rounded-xl h-12 w-32 flex text-2xl ${envelopesInputStyle}`}
            />
          </div>
        </div>
        <hr className="h-[2px] my-1 bg-gray-400/50 border-0 max-w-md mx-auto"></hr>
        <div
          className={`flex items-center justify-center w-full p-2 ${electorsEnvelopesDiffStyle}`}
        >
          <div className="flex p-2 justify-between items-center w-full  max-w-md ">
            <div
              className={`text-2xl text-neutral-700 font-bold px-3 py-5 tracking-wide ${electorsEnvelopesDiffStyle}`}
            >
              {electorsEnvelopesDiffStyle ? (
                <div className="flex flex-row gap-2">
                  Diferencia <img src="src/assets/icon/warn-icon.svg"></img>
                </div>
              ) : (
                'Diferencia'
              )}
            </div>
            <div className="text-2xl font-semibold px-3 py-5 mr-10">
              {electors - envelopes}
            </div>
          </div>
        </div>
        <div className="text-sm text-red max-w-md mx-auto text-left -mt-8 p-5">
          {electors - envelopes > 4
            ? 'La diferencia de votos no puede ser mayor a 4, esta mesa debe ser impugnada'
            : null}
        </div>
        <hr className="h-[2px] my-1 bg-gray-400/50 border-0 max-w-md mx-auto"></hr>
        {flatList.map((item, index) => (
          <div
            className="flex items-center justify-center my-6 w-full p-2"
            key={index}
          >
            <FlatList
              logo={item.logo}
              type={item.type}
              subTitle={item.subTitle}
              title={item.title}
              votes={item.votes}
              edit={item.edit}
              updateTotalVotes={updateTotalVotes}
            />
          </div>
        ))}
        <div className="flex items-center justify-center my-6 w-full p-2">
          <div className="flex p-2 justify-between items-center w-full  max-w-md">
            <div
              className={`text-3xl text-violet-brand font-bold px-3 py-5 tracking-wide ${totalVotesDiffStyle}`}
            >
              {totalVotesDiffStyle ? (
                <div className="flex flex-row gap-2">
                  Total <img src="src/assets/icon/warn-icon.svg"></img>
                </div>
              ) : (
                'Total'
              )}
            </div>
            <div
              className={`text-2xl font-semibold px-3 py-5 mr-10 ${totalVotesDiffStyle}`}
            >
              {totalVotes}
            </div>
          </div>
        </div>
        <div className="text-base text-red max-w-md mx-auto text-left -mt-16 p-5">
          {envelopes - totalVotes != 0
            ? 'El total de votos no coincide con la cantidad de sobres. Revisa los datos cargados'
            : null}
        </div>
        <div className="flex items-center justify-center text-sm my-10">
          <div className="flex items-center px-12">
            <div className="inline-flex items-center">
              <label
                className="relative flex items-center p-3 rounded-full cursor-pointer"
                data-ripple-dark="true"
              >
                <input
                  id="login"
                  type="checkbox"
                  checked={correctData}
                  onChange={handleCheckbox}
                  className="before:content[''] peer relative h-7 w-7 cursor-pointer appearance-none rounded-md border-2 border-violet-brand transition-all before:absolute before:top-2/4 before:left-2/4 before:block before:h-12 before:w-12 before:-translate-y-2/4 before:-translate-x-2/4 before:rounded-full before:bg-blue-gray-500 before:opacity-0 before:transition-opacity checked:border-violet-brand checked:bg-violet-brand checked:before:bg-violet-500 hover:before:opacity-10"
                />
                <div className="absolute text-white transition-opacity opacity-0 pointer-events-none top-2/4 left-2/4 -translate-y-2/4 -translate-x-2/4 peer-checked:opacity-100">
                  <img src="src/assets/icon/check-icon.svg" alt="check" />
                </div>
              </label>
            </div>
            <div className="px-3">
              <h3 className="text-start text-base">
                Verifico que controlé y que todos los datos son correctos.
              </h3>
            </div>
          </div>
        </div>
        <div className="flex items-center justify-center my-10">
          {0 <= electors - envelopes &&
          electors - envelopes <= 4 &&
          envelopes - totalVotes === 0 &&
          totalVotes != 0 &&
          correctData ? (
            <Link to="/send-success" className="w-full mx-6">
              <Button
                className="bg-violet-brand p-4 text-white rounded-xl font-semibold text-xl tracking-wider w-full"
                type="submit"
                label="Enviar Datos"
              />
            </Link>
          ) : (
            <div className="w-full mx-6">
              <Button
                className="bg-gray-300 p-4 text-black rounded-xl font-semibold text-xl tracking-wider w-full cursor-default"
                type="submit"
                label="Enviar Datos"
              />
            </div>
          )}
        </div>
        <div className="flex items-center justify-center my-10">
          <Link to="/" className="w-full mx-6">
            <Button
              className="text-red bg-transparent p-3 w-full rounded-xl text-xl"
              type="submit"
              label="Denunciar Irregularidad"
            />
          </Link>
        </div>
      </div>
    </section>
  );
};

export const LoadInformation = observer(LoadInformationPage);

export default LoadInformation;
