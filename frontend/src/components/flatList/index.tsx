import { useState } from 'react';
import { FlatListProps } from './types';

const FlatList = ({
  logo,
  type,
  subTitle,
  title,
  votes,
  edit = false,
  updateTotalVotes,
}: FlatListProps) => {
  const [vote, setVote] = useState<number>(votes);

  const handleVoteChange = (value: number) => {
    const newValue: number = value;
    if (newValue >= 0) {
      setVote(newValue);
      updateTotalVotes(newValue - vote);
    }
  };

  const titleColor: any = {
    massa: 'text-sky-400',
    milei: 'text-violet-800',
    null: 'text-neutral-500',
    appealed: 'text-neutral-500',
    contested: 'text-neutral-500',
    electoralCommand: 'text-neutral-500',
    blank: 'text-neutral-500',
  };

  const selectedInputStyle: string | null =
    vote > 0 ? 'border-2 border-violet-brand !text-black' : null;

  return (
    <div className="flex p-2 justify-between items-center w-full  max-w-md gap-4">
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
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleVoteChange(Number(e.target.value))}
        value={vote}
        readOnly={!edit}
        className={`border-2 text-center border-gray-300 outline-none cursor-default bg-white text-neutral-700 font-bold rounded-xl h-12 w-32 flex text-2xl ${selectedInputStyle}`}
      />
    </div>
  );
};

export default FlatList;
