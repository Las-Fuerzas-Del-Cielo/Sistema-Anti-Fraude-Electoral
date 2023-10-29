export function generateUniqueId(): string {
    const timestamp = new Date().getTime();
    const randomStr = Math.random().toString(36).substring(2, 15);
    return `${timestamp}-${randomStr}`;
  }
  