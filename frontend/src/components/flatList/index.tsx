import { useState } from 'react';
import { FlatListProps } from './types';

const FlatList = ({
  logo,
  type,
  subTitle,
  title,
  votes,
  edit = false,
}: FlatListProps) => {
  const [vote, setVote] = useState<number>(votes);

  const titleColor: any = {
    massa: 'text-sky-400',
    milei: 'text-purple-800',
    blank: 'text-neutral-700',
    noValidate: 'text-neutral-700',
    absent: 'text-neutral-700',
  };

  return (
    <div className="flex p-2 justify-between items-center w-full  max-w-md ">
      <img src={logo} alt="logo" className="w-16 h-16" />
      <div className="flex flex-col justify-start items-start mt-3">
        <label
          className={` ${titleColor[type]} text-xl font-bold leading-[15px]`}
        >
          {subTitle}
        </label>
        <label
          className={`text-neutral-700 mt-1 text-base font-semibold leading-7`}
        >
          {title}
        </label>
      </div>
      <input
        type="number"
        onChange={(e) => setVote(Number(e.target.value))}
        value={vote}
        readOnly={!edit}
        className={`border-2 text-center border-gray-300 outline-none cursor-default bg-white text-neutral-700 font-bold rounded-xl h-12 w-28 flex text-xl`}
      />
    </div>
  );
};

export default FlatList;
