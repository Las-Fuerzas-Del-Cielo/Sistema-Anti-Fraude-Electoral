import { getBase64 } from '#/utils';

// eslint-disable-next-line no-unused-vars
export function UploadImage({ onUpload }: { onUpload: (image: string) => void }) {
  async function onUploadInternal(file: File | null | undefined) {
    if (!file) return;
    const base64 = await getBase64(file);

    onUpload(base64);
  }

  return (
    <div
      className='flex flex-col items-center text-lg gap-16'
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.items[0].getAsFile();

        onUploadInternal(file);
      }}
    >
      <div className='flex items-center justify-center w-full'>
        <label
          className='flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer'
          htmlFor='dropzone-file'
        >
          <div className='flex flex-col items-center justify-center pt-5 pb-6 gap-2'>
            <img alt='' src='src/assets/icon/upload.svg' />
            <p className='mb-2 text-m'>Subir desde galería</p>
          </div>
          <input
            accept='image/*'
            className='hidden'
            id='dropzone-file'
            type='file'
            onChange={(ev) => onUploadInternal(ev.target.files?.[0])}
          />
        </label>
      </div>
      <label
        className='bg-purple-700 p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider'
        htmlFor='open-camera'
      >
        <input
          accept='image/*'
          capture='user'
          className='hidden'
          id='open-camera'
          type='file'
          onChange={(ev) => onUploadInternal(ev.target.files?.[0])}
        />
        Tomar foto
      </label>
    </div>
  );
}
