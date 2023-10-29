/* eslint-disable no-unused-vars */

enum Type {
  massa = 'massa',
  milei = 'milei',
  blank = 'blank',
  noValidate = 'noValidate',
  absent = 'absent',
}

export interface FlatListProps {
  logo?: React.ReactNode;
  subTitle: string;
  edit?: boolean;
  title?: string;
  type: Type;
  votes: number;
}
