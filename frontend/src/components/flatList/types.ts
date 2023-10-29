/* eslint-disable no-unused-vars */
export enum FlatListTypeEnum {
  massa = 'massa',
  milei = 'milei',
  blank = 'blank',
  null = 'null',
  noValidate = 'noValidate',
  absent = 'absent',
}

export interface FlatListProps {
  logo?: string;
  subTitle: string;
  edit?: boolean;
  title?: string;
  type: FlatListTypeEnum;
  votes: number;
}
