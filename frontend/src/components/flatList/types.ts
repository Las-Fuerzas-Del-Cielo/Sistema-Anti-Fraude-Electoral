export enum FlatListTypeEnum {
  massa = 'massa',
  milei = 'milei',
  blank = 'blank',
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
