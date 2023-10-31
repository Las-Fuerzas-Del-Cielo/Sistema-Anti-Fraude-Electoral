import { Request, Response, NextFunction } from 'express'
import { Session } from './models'
import { UploadedFile } from 'express-fileupload';

declare module 'express' {
  export type Middleware = (req: Request, res: Response, next: NextFunction) => void;

  interface Request {
    session?: Session;
    files?: {
      [key: string]: UploadedFile | UploadedFile[];
    };
  }
}
