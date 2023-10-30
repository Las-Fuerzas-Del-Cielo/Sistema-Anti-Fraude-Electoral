import { Request, Response } from 'express';
import { UploadedFile } from 'express-fileupload';
import path from 'path';

export const uploadFile = async (req: Request, res: Response): Promise<Response> => {
  if (!req.files || Object.keys(req.files).length === 0) {
    return res.status(400).json({ message: 'No files were uploaded.' });
  }

  const file: UploadedFile = req.files.file as UploadedFile;

  // Validate file type
  const allowedTypes: string[] = ['image/jpeg', 'image/png', 'image/gif'];
  if (!allowedTypes.includes(file.mimetype)) {
    return res.status(400).json({ message: 'Invalid file type.' });
  }

  // Validate file size (e.g., 5MB)
  const maxSize: number = 5 * 1024 * 1024; // 5MB in bytes
  if (file.size > maxSize) {
    return res.status(400).json({ message: 'File size exceeds limit.' });
  }

  // Save the file locally
  const fileName: string = file.name;
  const uploadPath: string = path.join(__dirname, '..', 'public', 'uploads', fileName);

  try {
    await new Promise<void>((resolve, reject) => {
      file.mv(uploadPath, (err: Error | null) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });

    return res.status(200).json({ message: 'File uploaded successfully.', fileName: fileName, filePath: uploadPath });
  } catch (err) {
    return res.status(500).json({ message: 'Error uploading file', error: err });
  }
};
