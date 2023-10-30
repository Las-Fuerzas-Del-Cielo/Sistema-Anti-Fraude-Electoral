const ImageInput = ({
  id = 'dropzone-file',
  handleOnChange,
  capture,
}: {
  id: string;
  handleOnChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  capture?: undefined | 'user' | 'environment';
}) => {
  return (
    <input
      id={id}
      type="file"
      className="hidden"
      accept="image/*"
      capture={capture}
      onChange={handleOnChange}
    />
  );
};

export default ImageInput;
