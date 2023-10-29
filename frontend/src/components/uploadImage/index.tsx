import { getBase64 } from "#/utils";
import { ChangeEvent } from "react";

export function UploadImage({
  onUpload,
}: {
  onUpload: (image: string) => void;
}) {
  async function onUploadInternal(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const base64 = await getBase64(file);
    onUpload(base64);
  }

  return (
    <div className="flex flex-col items-center text-lg gap-16">
      <div className="flex items-center justify-center w-full">
        <label
          htmlFor="dropzone-file"
          className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer"
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6 gap-2">
            <img src="src/assets/icon/upload.svg" alt="" />
            <p className="mb-2 text-m">Subir desde galería</p>
          </div>
          <input
            id="dropzone-file"
            type="file"
            className="hidden"
            accept="image/*"
            onChange={onUploadInternal}
          />
        </label>
      </div>
      <label
        htmlFor="open-camera"
        className="bg-purple-700 p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider"
      >
        <input
          id="open-camera"
          type="file"
          className="hidden"
          accept="image/*"
          capture="user"
          onChange={onUploadInternal}
        />
        Tomar foto
      </label>
    </div>
  );
}
