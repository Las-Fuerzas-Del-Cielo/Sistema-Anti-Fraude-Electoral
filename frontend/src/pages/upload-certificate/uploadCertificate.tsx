import { Link } from 'react-router-dom';
import { UploadImage } from '#/components/uploadImage';
import { useState } from 'react';
import { ProgressIndicator } from '#/components';
import { ProgressStepStatus } from '#/components/progressIndicator/types';
import './styles.css';

const CheckItem = ({ text }: { text: string }) => (
  <div className="flex justify-space-around items-center gap-2 h-12">
    <div className="flex justify-center items-center rounded-full bg-green-check text-white w-5 h-5 flex-shrink-0">
      <img className="w-3 h-3" src="src/assets/check-icono.svg" alt="" />
    </div>
    <p>{text}</p>
  </div>
);

const UploadCertificate = () => {
  // TODO: Replace with context useState
  const [, setCertificateImage] = useState<string>();

  return (
    <section className="bg-gray-100 items-center flex flex-col ">
      <div className="bg-violet-brand p-4 w-full flex flex-col justify-center items-center text-white ">
        <div className="w-full flex flex-row">
          <div className="flex flex-col justify-center items-center basis-1/4">
            <Link to="/">
              <img
                src="src/assets/images/back-arrow.svg"
                alt="back arrow"
                className="object-cover rounded w-6  sm:w-8  h-auto "
              />
            </Link>
          </div>
          <div className="basis-auto w-full flex justify-center items-center">
            <img
              src="src/assets/logos/fenix-white.svg"
              alt="fenix white"
              className="object-cover rounded w-35 sm:w-36 lg:w-36 h-auto w-image-25 "
            />
          </div>
        </div>

        <h1 className="text-4xl mb-5 mt-5 message">Tomar foto</h1>
      </div>
      <div className="p-4 w-full">
        <div className="container mx-auto flex-column my-210">
          <ProgressIndicator
            steps={[
              ProgressStepStatus.Active,
              ProgressStepStatus.Pending,
              ProgressStepStatus.Pending,
            ]}
          />
          <div className="text-start my-6">
            <p>
              Usá la cámara para subir el <b>certificado del fiscal</b>, o
              cargala desde la galería.
            </p>
            <div className="flex-column gap-2">
              <CheckItem text="Buscá un lugar con buena luz." />
              <CheckItem text="Asegurate de que se vean todos los datos." />
              <CheckItem
                text="Asegurate que el certificado esté firmado por el presidente de
                  tu mesa."
              />
            </div>
          </div>
        </div>
        <UploadImage onUpload={setCertificateImage} />
      </div>
    </section>
  );
};

export default UploadCertificate;
