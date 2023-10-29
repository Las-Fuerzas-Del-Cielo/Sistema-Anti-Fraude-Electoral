export const sleep = (ms: number | undefined) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });

export const getCurrentYear = () => {
  return new Date().getFullYear();
};
