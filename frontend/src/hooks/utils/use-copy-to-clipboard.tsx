import { useState } from 'react';

type CopiedValue = string | null;
// eslint-disable-next-line no-unused-vars
type CopyFn = (text: string) => Promise<boolean>;

export const useCopyToClipboard = (): [CopyFn, CopiedValue] => {
  const [copiedText, setCopiedText] = useState<CopiedValue>(null);

  const copy: CopyFn = async (text) => {
    if (!navigator?.clipboard) {
      return false;
    }

    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(text);

      return true;
    } catch (error) {
      setCopiedText(null);

      return false;
    }
  };

  return [copy, copiedText];
};

// usage :  const [value, copy] = useCopyToClipboard()
