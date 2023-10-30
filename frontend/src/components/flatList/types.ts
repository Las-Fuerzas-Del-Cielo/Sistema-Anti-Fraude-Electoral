/* eslint-disable no-unused-vars */
type UpdateTotalVotesFunction = (votesDifference: number) => void;

export enum FlatListTypeEnum {
  massa = 'massa',
  milei = 'milei',
  null = 'null',
  appealed = 'appealed',
  contested = 'contested',
  electoralCommand = 'electoralCommand',
  blank = 'blank',
}

export interface FlatListProps {
  logo?: string;
  subTitle: string;
  edit?: boolean;
  title?: string;
  type: FlatListTypeEnum;
  votes: number;
  updateTotalVotes: UpdateTotalVotesFunction;
}
