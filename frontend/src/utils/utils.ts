export const sleep = (ms: number | undefined) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });

export const getCurrentYear = () => {
  return new Date().getFullYear();
};

export function getBase64(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
      resolve(reader.result as string);
    };
    reader.onerror = function (error) {
      reject(error);
    };
  });
}
