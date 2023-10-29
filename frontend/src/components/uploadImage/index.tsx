import { getBase64 } from '#/utils';

export function UploadImage({
  onUpload,
}: {
  onUpload: (image: string) => void;
}) {
  async function onUploadInternal(file: File | null | undefined) {
    if (!file) return;
    const base64 = await getBase64(file);
    onUpload(base64);
  }

  return (
    <div
      className="flex flex-col items-center text-lg gap-16"
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.items[0].getAsFile();
        onUploadInternal(file);
      }}
    >
      <div className="flex items-center justify-center w-full">
        <label
          htmlFor="dropzone-file"
          className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer"
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6 gap-2">
            <img src="src/assets/icon/upload.svg" alt="" />
            <p className="mb-2 text-m">Subir una imagen de la galería</p>
          </div>
          <input
            id="dropzone-file"
            type="file"
            className="hidden"
            accept="image/*"
            onChange={(ev) => onUploadInternal(ev.target.files?.[0])}
          />
        </label>
      </div>
      <label
        htmlFor="open-camera"
        className="bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider"
      >
        <input
          id="open-camera"
          type="file"
          className="hidden"
          accept="image/*"
          capture="user"
          onChange={(ev) => onUploadInternal(ev.target.files?.[0])}
        />
        Tomar foto
      </label>
    </div>
  );
}
