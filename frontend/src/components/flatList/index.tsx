import { useState } from "react";
import { FlatListProps } from "./types";

const FlatList = ({
  logo,
  type,
  subTitle,
  title,
  votes,
  edit = false,
}: FlatListProps) => {
  const [vote, setVote] = useState<number>(votes);

  const titleColor = {
    massa: "text-sky-400",
    milei: "text-purple-800",
    blank: "text-neutral-700",
    noValidate: "text-neutral-700",
    absent: "text-neutral-700",
  };

  return (
    <div className="flex p-2 justify-between items-center w-full  max-w-md ">
      {logo && logo}
      <div className="flex flex-col justify-start items-start mt-3">
        <label
          className={` ${titleColor[type]} text-xl font-bold leading-[15px]`}
        >
          {subTitle}
        </label>
        {type === "noValidate" ? (
          <label
            className={`text-purple-800 mt-1   text-[10px] text-start font-normal leading-[15px]`}
          >
            {title?.split("\n").map((item, i) => (
              <span key={i}>
                {item}
                <br />
              </span>
            ))}
          </label>
        ) : (
          <label
            className={`text-neutral-700 mt-1  text-base font-semibold
          }  leading-[15px]`}
          >
            {title}
          </label>
        )}
      </div>
      <input
        type="number"
        onChange={(e) => setVote(Number(e.target.value))}
        value={vote}
        readOnly={!edit}
        className={`border-2 text-center  ${
          type === "noValidate" ? "border-neutral-200" : "border-purple-800"
        } outline-none cursor-default bg-white text-neutral-700 font-bold rounded-xl h-12  w-24 flex  text-xl`}
      />
    </div>
  );
};

export default FlatList;
