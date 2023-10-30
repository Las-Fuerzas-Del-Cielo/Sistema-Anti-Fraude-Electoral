import { toast, Toaster, ToastPosition } from 'react-hot-toast';
import { ToastiBarProps } from './types';

const toastiBar = ({
  text,
  action,
  close,
  twoLine,
  textTwo,
  actionText,
}: ToastiBarProps) => {
  let longerAction = false;

  if (actionText && actionText?.length > 7) {
    longerAction = true;
  }

  const config = {
    duration: close ? Infinity : 4000,
    position: 'top-center' as ToastPosition,
  };

  toast.custom(
    (t) => (
      <div
        className={`w-full max-w-xs text-white text-xs rounded-lg py-4 px-4 boun flex ${
          longerAction ? 'flex-col' : ''
        } justify-start bg-violet-brand`}
        style={{
          opacity: t.visible ? 1 : 0,
          transition: 'opacity 100ms ease-in-out',
        }}
      >
        <div className={`text-left flex-1 ${twoLine ? 'flex flex-col' : ''} `}>
          <p>{text}</p>
          {textTwo && <p>{textTwo}</p>}
        </div>
        <div
          className={`flex gap-3 ${
            longerAction ? 'w-full justify-end' : 'w-3/12'
          }`}
        >
          {action && (
            <div
              className={`${
                longerAction ? 'w-2/12' : 'w-3/4'
              } flex justify-center items-center`}
            >
              <p role="button" onClick={() => action()}>
                {actionText}
              </p>
            </div>
          )}
          {close && (
            <div
              className={`${
                longerAction ? 'w-1/12' : 'w-1/4'
              } flex justify-center items-center`}
            >
              <p role="button" onClick={() => toast.dismiss(t.id)}>
                <img
                  className="w-4 h-4"
                  src="/src/assets/icon/close.svg"
                  alt=""
                />
              </p>
            </div>
          )}
        </div>
      </div>
    ),
    config,
  );
};

export const SnackBar = (props: ToastiBarProps) => {
  toastiBar(props);
  return <Toaster />;
};
