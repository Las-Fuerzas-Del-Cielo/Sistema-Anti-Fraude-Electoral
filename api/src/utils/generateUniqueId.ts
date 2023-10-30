export function generateUniqueId(): string {
    const timestamp: number = new Date().getTime();
    const randomStr: string = Math.random().toString(36).substring(2, 15);
    return `${timestamp}-${randomStr}`;
  }
  